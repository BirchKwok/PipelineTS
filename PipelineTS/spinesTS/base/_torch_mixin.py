import copy
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from spinesUtils.asserts import ParameterValuesAssert, augmented_isinstance, ParameterTypeAssert, raise_if_not
from spinesUtils.logging import Logger

from PipelineTS.spinesTS.metrics import WMAPELoss, RMSELoss
from PipelineTS.spinesTS.utils import seed_everything, check_is_fitted

logger = Logger(with_time=False, name='')


class CQRWrapper(nn.Module):
    """Wraps any base nn.Module to produce 3 quantile predictions for CQR.

    Output shape: (B, 3 * out_features) laid out as [q_lower | q_median | q_upper].
    Uses softplus on learned offsets to guarantee q_lower <= median <= q_upper.
    """

    def __init__(self, base_model, out_features, alpha=0.1):
        super().__init__()
        self.base_model = base_model
        self.out_features = out_features
        self.alpha = alpha

        hidden = max(out_features * 2, 32)
        self.lower_head = nn.Sequential(
            nn.Linear(out_features, hidden), nn.GELU(),
            nn.Linear(hidden, out_features),
        )
        self.upper_head = nn.Sequential(
            nn.Linear(out_features, hidden), nn.GELU(),
            nn.Linear(hidden, out_features),
        )
        # Small init so initial intervals are tight around median
        with torch.no_grad():
            self.lower_head[-1].weight.mul_(0.1)
            self.lower_head[-1].bias.zero_()
            self.upper_head[-1].weight.mul_(0.1)
            self.upper_head[-1].bias.zero_()

    def forward(self, x):
        center = self.base_model(x)                        # (B, out_features)
        lower_offset = torch.nn.functional.softplus(self.lower_head(center))
        upper_offset = torch.nn.functional.softplus(self.upper_head(center))
        q_lower = center - lower_offset
        q_upper = center + upper_offset
        # (B, 3 * out_features): [lower | median | upper]
        return torch.cat([q_lower, center, q_upper], dim=-1)


class SinkhornResidualGate(nn.Module):
    """mHC-inspired residual gate with doubly-stochastic constraint.

    Inspired by DeepSeek's Manifold-Constrained Hyper-Connections (mHC),
    this module applies a learnable mixing gate between model output and
    a residual projection of the input. The gate matrix is projected onto
    the Birkhoff polytope via Sinkhorn-Knopp normalization, ensuring:
    - Spectral norm <= 1 (no signal amplification)
    - Closure under composition (stable across depth)
    - Smooth gradient flow

    Forward pass:
        residual = linear_proj(input)
        output = alpha * sinkhorn(W) @ model(input) + (1 - alpha) * residual

    Parameters
    ----------
    base_model : nn.Module
        The original model to wrap.
    in_features : int
        Input dimension (lags).
    out_features : int
        Output dimension (lags / forecast horizon).
    n_sinkhorn_iters : int, default=10
        Sinkhorn-Knopp iterations for doubly-stochastic projection.
    init_alpha : float, default=0.3
        Initial mixing weight for model output (conservative start).
    """

    def __init__(self, base_model, in_features, out_features,
                 n_sinkhorn_iters=10, init_alpha=0.3):
        super().__init__()
        self.base_model = base_model
        self.out_features = out_features
        self.n_sinkhorn_iters = n_sinkhorn_iters

        # Learnable mixing matrix (will be Sinkhorn-projected)
        # Initialize near identity for conservative start
        self.W_logits = nn.Parameter(
            torch.eye(out_features) * 2.0 + torch.randn(out_features, out_features) * 0.01
        )

        # Learnable residual scale (sigmoid -> [0, 1])
        # Initialize so sigmoid(raw) ≈ init_alpha
        init_logit = np.log(init_alpha / (1.0 - init_alpha + 1e-8))
        self.alpha_logit = nn.Parameter(torch.tensor(float(init_logit)))

        # Residual projection (input -> output dimension)
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def _sinkhorn(self, M):
        """Project matrix onto Birkhoff polytope (doubly stochastic).

        Iterative row/column normalization. The result has all rows
        and columns summing to 1, with spectral norm <= 1.
        """
        # Ensure non-negative via exp
        S = torch.exp(M)
        for _ in range(self.n_sinkhorn_iters):
            S = S / (S.sum(dim=1, keepdim=True) + 1e-8)  # Row normalize
            S = S / (S.sum(dim=0, keepdim=True) + 1e-8)  # Column normalize
        return S

    def forward(self, x):
        # Get model prediction
        model_out = self.base_model(x)  # (B, out_features)

        # Sinkhorn-normalized mixing matrix
        W = self._sinkhorn(self.W_logits)  # (out, out), doubly stochastic

        # Apply doubly-stochastic mixing to model output
        # This prevents any single output dimension from being amplified
        mixed = torch.matmul(model_out, W)  # (B, out_features)

        # Learnable residual blend
        alpha = torch.sigmoid(self.alpha_logit)
        residual = self.residual_proj(x)  # (B, out_features)

        return alpha * mixed + (1.0 - alpha) * residual


@ParameterTypeAssert({
    'device': (str, None)
})
def detect_available_device(device='auto'):
    device = device or 'auto'

    if device != 'auto':
        device = device.lower()
        auto_selector = False
    else:
        auto_selector = True

    tpu_available = False

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()

        if device is not None:
            tpu_available = True

    except ImportError:
        if device == 'tpu':
            logger.warning("[ImportError]: torch_xla package is not installed."
                           "Consider run `python3 -m pip install torch_xla` in your terminal.")

    mps_available = False
    cuda_available = False
    cpu_available = True
    mps_use = False
    cuda_use = False
    cpu_use = False
    tpu_use = False

    if torch.backends.mps.is_available():
        mps_available = True

    if torch.cuda.is_available():
        cuda_available = True

    if tpu_available and (auto_selector or device == 'tpu'):
        device = xm.xla_device()
        tpu_use = True
    elif cuda_available and (auto_selector or device == 'cuda'):
        device = 'cuda:0' if torch.cuda.device_count() > 1 else 'cuda'
        cuda_use = True
    elif mps_available and (auto_selector or device == 'mps'):
        device = 'mps'
        mps_use = True
    elif cpu_available and (auto_selector or device == 'cpu'):
        device = 'cpu'
        cpu_use = True

    # Only show available platforms
    lines = []
    if mps_available:
        lines.append(f"MPS  available: {mps_available}   | MPS  use: {mps_use}")
    if cuda_available:
        lines.append(f"CUDA available: {cuda_available}   | CUDA use: {cuda_use}")
    if tpu_available:
        lines.append(f"TPU  available: {tpu_available}   | TPU  use: {tpu_use}")
    if cpu_available:
        lines.append(f"CPU  available: {cpu_available}   | CPU  use: {cpu_use}")
    
    string_format = '\n'.join(lines)

    return device, string_format


def clear_torch_cache(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


@ParameterValuesAssert({
    'name': lambda s: augmented_isinstance(s, (None, str))
})
def get_loss_func(name=None):
    """get loss function

    Parameters
    ----------
    name: str, name of loss function, default None

    Returns
    -------
    object, loss function.
    """
    names = {'huber': nn.HuberLoss(), 'mse': nn.MSELoss(), 'mae': nn.L1Loss(),
             'wmape': WMAPELoss(), 'rmse': RMSELoss()}
    if isinstance(name, str):
        name = name.lower()
        return names[name]
    else:
        return names['mae']


class TorchModelMixin:
    """Provide pytorch models common mixin class.

    This class make it easy to write code like this:
        ```python
        class Model(TorchModelMixin):
            def __init__(self, *args, **kwargs):
                # need to set random seed if you needed
                # need to set device which to put your tensor, default to cuda/cuda:0 if your gpu is available, else to cpu
                super(Model, self).__init__(seed=None, device=None)

                self.model, self.loss_fn, self.optimizer = self.call()  # implement your model architecture

            def call(self):
                # model = your_model_class()
                # loss_fn = your_loss_function()
                # optimizer = your_optimizer_function()
                # return model, loss_fn, optimizer
                pass

            def fit(self, X, y, *args, **kwargs):
                return super().fit(X, y, *args, **kwargs)

            # def metric(self, y_true, y_pred):
                 # your metric, default to mae(mean absolute error)
                 # if you want to use other metrics, you need to override this function

        # To fit something
        model = Model(args, kwargs)
        model.fit(X, y)

        # To predict something
        y_pred = model.predict(X)
        ```

    Parameters
    ----------
    seed : int or None, random seed
    device : str or None, device name
    loss_fn: str or None, loss function

    Returns
    -------
    None

    """

    def __init__(self, seed=None, device='auto', loss_fn='mae') -> None:
        self.training_logs = {
            'time_cost': [],
            'epochs': [],
            'batches': [],
            'lrs': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'current_p': []
        }

        seed_everything(seed)
        self.device, self.string_format = detect_available_device(device)

        clear_torch_cache(self.device)

        self.loss_fn_name = loss_fn
        self.loss_fn = get_loss_func(loss_fn)
        self.current_patience = 0
        self.model, self.loss_fn, self.optimizer = self.call()

        self.__spinesTS_is_fitted__ = False

    def call(self, *args, **kwargs):
        """To implement the model architecture.

        """
        raise NotImplementedError("To implement a spinesTS.nn model class, you must implement a call function.")

    def fit(self,
            X,
            y,
            epochs=3000,
            batch_size='auto',
            eval_set=None,
            loss_type='down',
            metrics_name='score',
            monitor='val_loss',
            min_delta=0,
            patience=10,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.1,
            restore_best_weights=True,
            verbose=True,
            use_ema=False,
            ema_decay=0.999,
            use_swa=False,
            swa_start_frac=0.75,
            warmup_epochs=0,
            **lr_scheduler_kwargs
            ):
        """Fit your model.

        Parameters
        ----------
        X : torch.Tensor, training features
        y : torch.Tensor, training targets
        epochs : int, training epochs, default to 1000
        batch_size : str or int, 'auto' means to autofit, int means to specify the batch size
        eval_set : iterable object(tuple or list) of torch.Tensor, default to None
        loss_type : str, 'down' or 'rise', only be used if lr_scheduler='ReduceLROnPlateau'
            it means the way to set the learning rate scheduler to watch the loss value (down or rise)
        metrics_name : str, names your metrics, default to 'score'
        monitor : str, 'val_loss' or 'loss', quantity to be monitored,
        min_delta : minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement, default to 0
        patience : number of epochs with no improvement after which training will be stopped, default to 10
        lr_scheduler : learning rate scheduler name, one of ['ReduceLROnPlateau', 'CosineAnnealingLR',
            'CosineAnnealingWarmRestarts', None]
        lr_scheduler_patience :  number of epochs with no improvement after which learning rate will be reduced.
            For example, if patience = 2, then we will ignore the first 2 epochs with no improvement,
            and will only decrease the LR after the 3rd epoch  if the loss still hasn’t improved then, default: 10
        lr_factor : factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1
        restore_best_weights : Whether to restore model weights
                        from the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of training are used.
            If True, and if no epoch improves, training will run for patience epochs and restore weights from
                the best epoch in that set. Default to True.
        verbose : Whether to  displays messages, default to True
        **lr_scheduler_kwargs : torch.optim.lr_scheduler parameters

        Returns
        -------
        self

        """
        return self._fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            eval_set=eval_set,
            loss_type=loss_type,
            metrics_name=metrics_name,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            lr_scheduler=lr_scheduler,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
            use_ema=use_ema,
            ema_decay=ema_decay,
            use_swa=use_swa,
            swa_start_frac=swa_start_frac,
            warmup_epochs=warmup_epochs,
            **lr_scheduler_kwargs
        )

    def _enable_cqr(self, alpha=0.1):
        """Enable Conformalized Quantile Regression mode.

        Wraps ``self.model`` with :class:`CQRWrapper`, switches the loss to
        :class:`CombinedQuantileLoss`, and rebuilds the optimizer to include
        the new quantile-head parameters.
        """
        from PipelineTS.spinesTS.metrics import CombinedQuantileLoss

        self.model = CQRWrapper(self.model, self.out_features, alpha=alpha)
        self.loss_fn = CombinedQuantileLoss(alpha=alpha)
        weight_decay = getattr(self, 'weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self._cqr_enabled = True
        self._cqr_alpha = alpha

    def _enable_residual_gate(self, n_sinkhorn_iters=10, init_alpha=0.3):
        """Enable mHC-inspired Sinkhorn residual gate.

        Wraps ``self.model`` with :class:`SinkhornResidualGate`, which adds a
        doubly-stochastic mixing matrix (Birkhoff polytope projection) and a
        learnable residual blend. This prevents signal amplification and
        stabilizes training, especially for noisy or oscillating loss.

        Must be called before ``_fit()`` (after model is defined).
        """
        self.model = SinkhornResidualGate(
            self.model,
            in_features=self.in_features,
            out_features=self.out_features,
            n_sinkhorn_iters=n_sinkhorn_iters,
            init_alpha=init_alpha,
        )
        weight_decay = getattr(self, 'weight_decay', 1e-4)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        self._residual_gate_enabled = True

    def predict(self, X):
        """
        X : torch.Tensor or np.ndarray, data to predict.
            Supports both 2D (batch, seq_len) for univariate
            and 3D (batch, seq_len, n_vars) for multivariate.
        """
        check_is_fitted(self)
        self.model.eval()
        with torch.inference_mode():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            elif not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X).float()
            X = self._move_to_device(X)
            pred = self.model(X)
        return pred.cpu().numpy()

    def _move_to_device(self, obj):
        obj = obj.to(self.device)
        return obj

    def metric(self, y_true, y_pred):
        """model metric"""
        if getattr(self, '_cqr_enabled', False) and y_pred.shape[-1] != y_true.shape[-1]:
            # Detect which argument is the CQR output (3x features)
            if y_pred.shape[-1] > y_true.shape[-1]:
                f = y_true.shape[-1]
                y_pred = y_pred[..., f:2 * f]
            else:
                f = y_pred.shape[-1]
                y_true = y_true[..., f:2 * f]
        return nn.functional.l1_loss(y_true, y_pred).item()

    def _get_batch_size(self, x, batch_size='auto'):
        if batch_size == 'auto':
            n = len(x)
            # Use power-of-2 batch size, capped between 16 and 512
            log_n = max(4, min(9, int(np.log2(max(n, 16)))))
            self._batch_size = int(2 ** log_n)
            self._batch_size = min(self._batch_size, n)
        else:
            raise_if_not(ValueError, isinstance(batch_size, int) and batch_size > 0, "batch_size must be int and > 0")
            self._batch_size = batch_size

    @staticmethod
    def _check_x_y_type(X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        elif not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        elif not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)

        return X.float(), y.float()

    def data_loader(self, X, y, shuffle=False):
        train_data = TensorDataset(X, y)
        train_loader = DataLoader(
            train_data, batch_size=self._batch_size, shuffle=shuffle,
            pin_memory=True if str(self.device).startswith('cuda') else False
        )

        return train_loader

    @staticmethod
    def _collect_gtb_aux_loss(model):
        """Collect auxiliary load-balancing losses from all GTB modules (MoE routing)."""
        aux = torch.tensor(0.0)
        for m in model.modules():
            if m.__class__.__name__ == 'GlobalTemporalBlock' and hasattr(m, '_aux_loss'):
                loss = m._aux_loss
                if loss is not None and loss.requires_grad:
                    aux = aux.to(loss.device) + loss
        return aux

    def train_on_one_epoch(
            self,
            dataloader,
            model,
            loss_fn,
            optimizer
    ):
        """Training function on one epoch
        If you want to override it, you just need to return two values,
        current loss on this epoch, average-accuracy on this epoch
        """
        model.train()  # set model to training mode
        train_batch = len(dataloader)
        train_loss_current = 0
        valid_batches = 0

        use_amp = hasattr(self, '_use_amp') and self._use_amp

        for x, y in dataloader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # clear optimizer gradient

            if use_amp:
                with torch.amp.autocast('cuda'):
                    train_pred = model(x)
                    train_loss = loss_fn(train_pred, y)
                    train_loss = train_loss + self._collect_gtb_aux_loss(model)

                # NaN guard: skip this batch if loss is NaN
                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    continue

                self._grad_scaler.scale(train_loss).backward()
                self._grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
                self._grad_scaler.step(optimizer)
                self._grad_scaler.update()
            else:
                # compute error
                train_pred = model(x)
                train_loss = loss_fn(train_pred, y)
                train_loss = train_loss + self._collect_gtb_aux_loss(model)

                # NaN guard: skip this batch if loss is NaN
                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    continue

                # backward
                train_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)

                optimizer.step()

            train_loss_current += train_loss.item()
            valid_batches += 1

        if valid_batches == 0:
            # All batches had NaN loss — restore best weights to recover
            if hasattr(self, 'best_weight') and self.best_weight is not None:
                self.model.load_state_dict(self.best_weight)
            return float('inf'), float('inf')

        avg_loss = train_loss_current / valid_batches
        return avg_loss, avg_loss

    def test_on_one_epoch(
            self,
            dataloader,
            model,
            loss_fn
    ):
        """
        Test function on one epoch
        If you want to override it, you just need to return two values,
        current loss on this epoch, average-accuracy on this epoch

        """

        model.eval()  # set model to evaluate mode
        test_loss, test_num_batches = 0, len(dataloader)
        with torch.inference_mode():
            for x, y in dataloader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                pred = model(x)
                test_loss += loss_fn(y, pred).item()  # scalar

        test_loss /= test_num_batches

        return test_loss, test_loss

    @ParameterTypeAssert({
        'loss_type': str,
        'min_delta': int,
        'patience': int,
        'restore_best_weights': bool
    })
    @ParameterValuesAssert({
        'loss_type': ('min', 'max')
    })
    def _early_stopping(
            self,
            loss,
            loss_type='min',
            min_delta=0,
            patience=10,
            restore_best_weights=True
    ):
        """
        loss type : min or max
        """
        if loss_type == 'max':
            loss = -loss

        if loss < (self.current_loss + min_delta):
            self.current_loss = loss
            if restore_best_weights:
                self.best_weight = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.current_patience = 0
        else:
            self.current_patience += 1

        if self.current_patience == patience:
            if restore_best_weights:
                self.model.load_state_dict(self.best_weight)
            return True

        return False

    def _get_lr_scheduler(
            self, mode=None,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.1,
            epochs=100,
            steps_per_epoch=1,
            **lr_scheduler_kwargs
    ):
        # Remove non-scheduler kwargs that may leak in
        lr_scheduler_kwargs.pop('scheduler_config', None)

        if lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                patience=lr_scheduler_patience, factor=lr_factor, **lr_scheduler_kwargs
            )
        elif lr_scheduler == 'CosineAnnealingLR':
            # T_max should adapt to training duration
            t_max = max(5, epochs // 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=1e-7,
                **lr_scheduler_kwargs
            )
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            t_0 = max(5, epochs // 20)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=t_0, T_mult=2,
                **lr_scheduler_kwargs
            )
        elif lr_scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
                **lr_scheduler_kwargs
            )
        elif lr_scheduler is None:
            return None
        else:
            raise KeyError(f"{lr_scheduler} is invalid.")

        return scheduler

    def _print_training_log(self, init_lr, metrics_name, total_epochs):
        training_log_print = OrderedDict({
            "epoch_msg": "",
            "current_p_msg": "",
            "lr_msg": "",
            "loss_msg": "",
            "val_loss_msg": "",
            "time_msg": ""
        })
        training_log_print["epoch_msg"] = f"Epoch {self.training_logs['epochs'][-1] + 1:>1d}/" + \
                                          f"{total_epochs:>1d}  " + \
                                          f"\n\r{self.training_logs['batches'][-1]}/{self.training_logs['batches'][-1]}"

        training_log_print["lr_msg"] = f"[*lr: {self.training_logs['lrs'][-1]:4e}]" if len(
            str(self.training_logs['lrs'][-1])) <= 5 else f" [*lr: {self.training_logs['lrs'][-1]:>5}]"

        training_log_print["loss_msg"] = f"loss: {self.training_logs['train_loss'][-1]:>.4f} - " + \
                                         f"{metrics_name}: {self.training_logs['train_accuracy'][-1]:>.4f}"

        if len(self.training_logs['test_loss']) == 0:
            del training_log_print['val_loss_msg']
        else:
            training_log_print[
                "val_loss_msg"
            ] = f"val_loss: {self.training_logs['test_loss'][-1]:>.4f} - " + \
                f"val_{metrics_name}: {self.training_logs['test_accuracy'][-1]:>.4f}"

        training_log_print["time_msg"] = \
            f"{self.training_logs['time_cost'][-1]:>.2f}s/epoch - " + \
            f"{self.training_logs['time_cost'][-1] / self.training_logs['batches'][-1]:>.3f}s/step"

        if len(self.training_logs['current_p']) == 0:
            del training_log_print['current_p_msg']
        else:
            training_log_print["current_p_msg"] = f"p{self.training_logs['current_p'][-1]}"

        if round(self.training_logs['lrs'][-1], 5) == init_lr:
            del training_log_print['lr_msg']
        metric_string = ' - '.join([i for i in training_log_print.values()])

        return metric_string

    @staticmethod
    def _check_eval_set_params(eval_set):
        if isinstance(eval_set, tuple):
            raise_if_not(ValueError, len(eval_set) == 2, "eval_set must be tuple of (X, y)")
        elif isinstance(eval_set, list):
            raise_if_not(ValueError, len(eval_set) == 1, "the length of eval_set must be 1")
            raise_if_not(TypeError, isinstance(eval_set[0], tuple), "the first element of eval_set must be tuple")
            raise_if_not(ValueError, len(eval_set[0]) == 2, "eval_set must be list of [X, y]")
            eval_set = eval_set[0]

        return eval_set

    def _fit(
            self,
            X,
            y,
            epochs=1000,
            batch_size='auto',
            eval_set=None,
            loss_type='min',
            metrics_name='score',
            monitor='val_loss',
            min_delta=0,
            patience=10,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.1,
            restore_best_weights=True,
            verbose=True,
            use_ema=False,
            ema_decay=0.999,
            use_swa=False,
            swa_start_frac=0.75,
            warmup_epochs=0,
            **lr_scheduler_kwargs
    ):
        """
        lr_scheduler: torch.optim.lr_scheduler class,
            only support to ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
        """

        raise_if_not(ValueError, isinstance(epochs, int) and epochs > 0, "epochs must be int and > 0")
        raise_if_not(ValueError, isinstance(batch_size, (int, str)), "batch_size must be int or str")
        raise_if_not(ValueError, isinstance(loss_type, str), "loss_type must be str")
        raise_if_not(ValueError, isinstance(metrics_name, str), "metrics_name must be str")
        raise_if_not(ValueError, isinstance(min_delta, int), "min_delta must be int")
        raise_if_not(ValueError, isinstance(patience, int), "patience must be int")
        raise_if_not(ValueError, isinstance(lr_scheduler, (str, type(None))), "lr_scheduler must be str or None")
        raise_if_not(ValueError, isinstance(lr_scheduler_patience, int), "lr_scheduler_patience must be int")
        raise_if_not(ValueError, isinstance(lr_factor, float), "lr_factor must be float")
        raise_if_not(ValueError, isinstance(restore_best_weights, bool), "restore_best_weights must be bool")
        raise_if_not(ValueError, isinstance(verbose, bool), "verbose must be bool")
        raise_if_not(ValueError, isinstance(lr_scheduler_kwargs, dict), "lr_scheduler_kwargs must be dict")
        raise_if_not(ValueError, monitor in ('train_loss', 'val_loss', None),
                     "monitor must be one of ['train_loss', 'val_loss', None]")
        raise_if_not(ValueError, isinstance(eval_set, (list, tuple, type(None))), "eval_set must be list or tuple")

        init_lr = copy.deepcopy(self.learning_rate)

        self.model = self._move_to_device(self.model)
        X, y = self._check_x_y_type(X, y)

        eval_set = self._check_eval_set_params(eval_set)

        self._get_batch_size(X, batch_size=batch_size)
        train_dataloader = self.data_loader(X, y, shuffle=True)
        test_dataloader = None

        # Enable AMP for CUDA devices
        self._use_amp = str(self.device).startswith('cuda')
        if self._use_amp:
            self._grad_scaler = torch.amp.GradScaler('cuda')

        if eval_set is not None:
            test_dataloader = self.data_loader(*self._check_x_y_type(eval_set[0], eval_set[1]))

        self.current_loss = np.finfo(np.float64).max - min_delta
        self.best_weight = {k: v.clone() for k, v in self.model.state_dict().items()}

        batches = int(np.ceil(len(X) / self._batch_size))

        mode = loss_type

        scheduler = self._get_lr_scheduler(
            mode=mode, lr_scheduler=lr_scheduler,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            epochs=epochs,
            steps_per_epoch=batches,
            **lr_scheduler_kwargs)

        # --- EMA: Exponential Moving Average of weights ---
        ema_state = None
        if use_ema:
            ema_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # --- SWA: Stochastic Weight Averaging ---
        swa_state = None
        swa_count = 0
        swa_start_epoch = int(epochs * swa_start_frac) if use_swa else epochs + 1

        # --- Warmup: linear LR warmup ---
        warmup_epochs = max(0, int(warmup_epochs))
        if warmup_epochs > 0:
            warmup_start_lr = init_lr * 0.1
            warmup_delta = (init_lr - warmup_start_lr) / max(warmup_epochs, 1)

        for epoch in range(epochs):
            tik = time.time()
            stop_state = False

            self.training_logs['epochs'].append(epoch)
            self.training_logs['batches'].append(batches)

            # --- Warmup LR override ---
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_lr = warmup_start_lr + warmup_delta * epoch
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            train_loss_current, train_acc = self.train_on_one_epoch(train_dataloader, model=self.model,
                                                                    loss_fn=self.loss_fn,
                                                                    optimizer=self.optimizer)

            if lr_scheduler and epoch >= warmup_epochs:
                scheduler.step() if lr_scheduler != 'ReduceLROnPlateau' else scheduler.step(train_loss_current)

            # --- EMA update ---
            if ema_state is not None:
                with torch.no_grad():
                    for k, v in self.model.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)

            # --- SWA accumulation ---
            if use_swa and epoch >= swa_start_epoch:
                sd = self.model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.clone() for k, v in sd.items()}
                else:
                    for k in swa_state:
                        swa_state[k] += sd[k]
                swa_count += 1

            self.training_logs['lrs'].append(round(float(self.optimizer.state_dict()['param_groups'][0]['lr']), 7))

            self.training_logs['train_loss'].append(train_loss_current)
            self.training_logs['train_accuracy'].append(train_acc)

            if monitor == 'train_loss':
                stop_state = self._early_stopping(train_loss_current, loss_type=loss_type,
                                                  min_delta=min_delta, patience=patience,
                                                  restore_best_weights=restore_best_weights)
            else:
                if test_dataloader:
                    test_loss, test_acc = self.test_on_one_epoch(test_dataloader, self.model, self.loss_fn)
                    stop_state = self._early_stopping(test_loss, loss_type=loss_type, min_delta=min_delta,
                                                      patience=patience, restore_best_weights=restore_best_weights)

                    self.training_logs['test_loss'].append(test_loss)
                    self.training_logs['test_accuracy'].append(test_acc)

            if monitor is not None and eval_set is not None:
                self.training_logs['current_p'].append(self.current_patience)

            tok = time.time()

            self.training_logs['time_cost'].append(tok - tik)

            if verbose:
                print(self._print_training_log(init_lr, metrics_name, epochs))

            if stop_state:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        # --- Apply weight averaging strategies ---
        # Priority: SWA > EMA > best_weight (from early stopping)
        if use_swa and swa_state is not None and swa_count > 0:
            for k in swa_state:
                swa_state[k] /= swa_count
            self.model.load_state_dict(swa_state)
        elif use_ema and ema_state is not None:
            self.model.load_state_dict(ema_state)

        self.__spinesTS_is_fitted__ = True
        return self

    def score(self, X, y):
        self.model.eval()
        with torch.inference_mode():
            X, y = torch.Tensor(X), torch.Tensor(y)
            X_gpu = self._move_to_device(X)
            pred = self.model(X_gpu).cpu()

        return self.metric(y, pred)
