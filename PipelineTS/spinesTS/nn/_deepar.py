"""DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.

Modern implementation using RWKV (linear RNN) encoder instead of traditional
LSTM for O(T) parallel temporal mixing, combined with a Gaussian probabilistic
output head. During training, the model learns to predict distribution
parameters (μ, σ) at each future timestep. At inference, point predictions
use the learned mean.

Architecture:
    Input → RevIN → Per-timestep Embedding → RWKV Encoder → Gated Residual
    Blocks → Gaussian Head (μ, σ) → RevIN Denormalize

Key features:
- RWKV encoder: all nn.Linear ops, no sequential recurrence, GPU-friendly
- Probabilistic training via Gaussian NLL loss
- RevIN normalization and direct residual shortcut
- Supports multivariate via MultivariateWrapper

Reference:
    Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive
    Recurrent Networks", International Journal of Forecasting, 2020.
"""

from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers import MultivariateWrapper
from PipelineTS.spinesTS.layers import RWKVEncoder


class GaussianHead(nn.Module):
    """Predicts Gaussian distribution parameters (mu, sigma) for each step.

    Uses a shared trunk with separate heads for mean and scale.
    Sigma is guaranteed positive via softplus.

    Args:
        d_model: Input feature dimension.
        out_features: Number of output timesteps.
    """

    def __init__(self, d_model, out_features):
        super().__init__()
        self.mu_head = nn.Linear(d_model, out_features)
        self.sigma_head = nn.Linear(d_model, out_features)
        # Initialize sigma head with small weights for tight initial variance
        with torch.no_grad():
            self.sigma_head.weight.mul_(0.1)
            self.sigma_head.bias.fill_(0.5)

    def forward(self, h):
        """
        Args:
            h: (B, d_model)
        Returns:
            mu: (B, out_features)
            sigma: (B, out_features), strictly positive
        """
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6
        return mu, sigma


class GatedResBlock(nn.Module):
    """Gated residual block with SiLU activation.

    x → LayerNorm → [sigmoid(gate) * SiLU(up)] → dropout → + residual
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.up = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        return x + self.drop(torch.sigmoid(self.gate(h)) * F.silu(self.up(h)))


class DeepARBlock(nn.Module):
    """DeepAR backbone with RWKV encoder and Gaussian output head.

    Embeds each timestep, processes with stacked RWKV blocks for temporal
    mixing, refines with gated residual blocks, then produces distribution
    parameters. Includes RevIN and a direct residual shortcut.

    Args:
        in_features: Input sequence length (lookback window).
        out_features: Prediction horizon.
        d_model: Hidden dimension.
        n_blocks: Number of gated residual refinement blocks.
        n_rwkv_blocks: Number of RWKV temporal mixing blocks.
        dropout: Dropout rate.
    """

    def __init__(self, in_features, out_features, d_model=64, n_blocks=3,
                 n_rwkv_blocks=3, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = 1e-5

        # Per-timestep embedding
        self.step_embed = nn.Linear(1, d_model)

        # RWKV encoder: linear temporal mixing
        self.rwkv = RWKVEncoder(
            seq_len=in_features, d_model=d_model,
            n_blocks=n_rwkv_blocks, expand_ratio=2.0,
            dropout=dropout
        )

        # Context vector: attention-weighted pooling over temporal positions
        self.attn_weight = nn.Linear(d_model, 1)

        # Gated residual refinement blocks
        self.blocks = nn.ModuleList([
            GatedResBlock(d_model, dropout)
            for _ in range(n_blocks)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        # Gaussian output head
        self.gaussian_head = GaussianHead(d_model, out_features)

        # Direct residual shortcut
        self.residual_proj = nn.Linear(in_features, out_features)

        # Training flag: when True, return (mu, sigma) for NLL loss
        self._return_distribution = True

    def forward(self, x):
        # x: (B, L) or (B, L, 1)
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN: instance normalization
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        x_norm = (x - mean) / std

        B, L = x_norm.shape

        # (B, L) -> (B, L, 1) -> (B, L, d_model)
        h = self.step_embed(x_norm.unsqueeze(-1))

        # RWKV temporal mixing
        h = self.rwkv(h)  # (B, L, d_model)

        # Attention-weighted pooling: learn which timesteps matter most
        attn_scores = self.attn_weight(h).squeeze(-1)  # (B, L)
        attn_probs = torch.softmax(attn_scores, dim=1)  # (B, L)
        context = torch.einsum('bl,bld->bd', attn_probs, h)  # (B, d_model)

        # Gated residual refinement
        for block in self.blocks:
            context = block(context)
        context = self.output_norm(context)

        # Gaussian head: predict distribution parameters
        mu, sigma = self.gaussian_head(context)

        # Add residual shortcut to mean
        mu = mu + self.residual_proj(x_norm)

        # RevIN denormalize
        mu = mu * std + mean
        sigma = sigma * std  # scale sigma by input std

        if self._return_distribution:
            # Return concatenated [mu | sigma] for the loss function
            return torch.cat([mu, sigma], dim=-1)  # (B, 2 * out_features)
        else:
            return mu


class GaussianNLLLossFn(nn.Module):
    """Gaussian negative log-likelihood loss for DeepAR.

    Given predictions (mu, sigma) and targets y, computes:
        NLL = 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]

    The input pred has shape (B, 2*H) where first H columns are mu,
    last H columns are sigma.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Handle argument order: whichever has 2x columns is the prediction
        if pred.shape[-1] != target.shape[-1]:
            if pred.shape[-1] > target.shape[-1]:
                # pred is the model output
                pass
            else:
                # arguments are swapped
                pred, target = target, pred

        H = target.shape[-1]
        mu = pred[..., :H]
        sigma = pred[..., H:2 * H]

        # Gaussian NLL
        variance = sigma ** 2 + 1e-8
        nll = 0.5 * (torch.log(variance) + (target - mu) ** 2 / variance)
        return nll.mean()


class DeepAR(TorchModelMixin, ForecastingMixin):
    """DeepAR model: probabilistic time series forecasting.

    Uses RWKV (linear RNN) encoder with Gaussian output head for
    probabilistic forecasting. Point predictions use the learned mean.

    Parameters
    ----------
    in_features : int
        Input sequence length (lookback window).
    out_features : int
        Prediction horizon.
    n_vars : int
        Number of input variables (1 for univariate).
    d_model : int
        Hidden dimension size.
    n_blocks : int
        Number of gated residual refinement blocks.
    n_rwkv_blocks : int
        Number of RWKV temporal mixing blocks.
    dropout : float
        Dropout rate.
    loss_fn : str
        Loss function name (overridden to Gaussian NLL internally).
    learning_rate : float
        Learning rate for optimizer.
    random_seed : int
        Random seed for reproducibility.
    device : str
        Compute device ('auto', 'cpu', 'cuda', 'mps').
    weight_decay : float
        L2 regularization strength.
    channel_mixing : bool
        Whether to apply cross-variable mixing in multivariate mode.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_vars: int = 1,
                 d_model: int = 64,
                 n_blocks: int = 3,
                 n_rwkv_blocks: int = 3,
                 dropout=0.1,
                 loss_fn='mae',
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 weight_decay: float = 1e-4,
                 channel_mixing: bool = True
                 ) -> None:
        self.in_features, self.out_features = in_features, out_features
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_rwkv_blocks = n_rwkv_blocks
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing

        # this sentence needs to be the last one
        super(DeepAR, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = DeepARBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            n_rwkv_blocks=self.n_rwkv_blocks,
            dropout=self.dropout
        )
        if self.n_vars > 1:
            model = MultivariateWrapper(
                backbone, self.n_vars, self.out_features,
                channel_mixing=self.channel_mixing
            )
        else:
            model = backbone

        # Use Gaussian NLL loss for probabilistic training
        loss_fn = GaussianNLLLossFn()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return model, loss_fn, optimizer

    def _enable_residual_gate(self, n_sinkhorn_iters=10, init_alpha=0.3):
        """Enable residual gate: disable Gaussian output first.

        The SinkhornResidualGate expects output dim = out_features, but
        the Gaussian head outputs 2*out_features ([mu|sigma]). Switch to
        point-prediction mode (mu only) and use MAE loss instead.
        """
        backbone = self._get_backbone()
        backbone._return_distribution = False
        self.loss_fn = torch.nn.L1Loss()
        super()._enable_residual_gate(
            n_sinkhorn_iters=n_sinkhorn_iters, init_alpha=init_alpha
        )

    def _enable_cqr(self, alpha=0.1):
        """Enable CQR mode: disable Gaussian output, use CQR for intervals.

        When CQR is enabled, the backbone switches to point-prediction mode
        (mu only), and CQR wraps it to produce quantile intervals.
        """
        # Switch backbone to point prediction mode first
        backbone = self._get_backbone()
        backbone._return_distribution = False
        # Switch loss to MAE for CQR training
        self.loss_fn = torch.nn.L1Loss()
        # Now call parent CQR enablement
        super()._enable_cqr(alpha=alpha)

    def predict(self, X):
        """Predict point estimates (mu) from the Gaussian head.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input data of shape (B, L) or (B, L, n_vars).

        Returns
        -------
        np.ndarray
            Point predictions of shape (B, out_features).
        """
        from PipelineTS.spinesTS.utils import check_is_fitted
        check_is_fitted(self)
        self.model.eval()

        # If CQR is enabled, parent predict handles everything
        if getattr(self, '_cqr_enabled', False):
            return super().predict(X)

        # Otherwise, temporarily disable distribution output for point predictions
        backbone = self._get_backbone()
        original_flag = backbone._return_distribution
        backbone._return_distribution = False

        with torch.inference_mode():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            elif not isinstance(X, torch.Tensor):
                X = torch.as_tensor(X).float()
            X = self._move_to_device(X)
            pred = self.model(X)

        backbone._return_distribution = original_flag
        return pred.cpu().numpy()

    def _get_backbone(self):
        """Get the DeepARBlock backbone, unwrapping any wrappers."""
        model = self.model
        # Unwrap SinkhornResidualGate
        if isinstance(model, SinkhornResidualGate):
            model = model.base_model
        # Unwrap CQRWrapper
        if isinstance(model, CQRWrapper):
            model = model.base_model
        # Unwrap MultivariateWrapper
        if isinstance(model, MultivariateWrapper):
            model = model.backbone
        return model

    def metric(self, y_true, y_pred):
        """Model metric: use MAE on the mu portion of the output."""
        if y_pred.shape[-1] != y_true.shape[-1]:
            # Extract mu from [mu | sigma] concatenation
            if y_pred.shape[-1] > y_true.shape[-1]:
                H = y_true.shape[-1]
                y_pred = y_pred[..., :H]
            else:
                H = y_pred.shape[-1]
                y_true = y_true[..., :H]
        # Also handle CQR output [q_lower | q_median | q_upper]
        if getattr(self, '_cqr_enabled', False) and y_pred.shape[-1] != y_true.shape[-1]:
            if y_pred.shape[-1] > y_true.shape[-1]:
                f = y_true.shape[-1]
                y_pred = y_pred[..., f:2 * f]
            else:
                f = y_pred.shape[-1]
                y_true = y_true[..., f:2 * f]
        return nn.functional.l1_loss(y_true, y_pred).item()

    def fit(self,
            X_train: Any,
            y_train: Any,
            epochs: int = 1000,
            batch_size: Union[str, int] = 'auto',
            eval_set: Any = None,
            monitor: str = 'val_loss',
            min_delta: int = 0,
            patience: int = 10,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.7,
            restore_best_weights: bool = True,
            verbose: bool = True,
            loss_type='min',
            **kwargs: Any) -> Any:
        return super().fit(X_train, y_train, epochs, batch_size, eval_set,
                           loss_type=loss_type,
                           metrics_name=self.loss_fn_name,
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience,
                           restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)


# Import wrappers at module level for isinstance checks in _get_backbone
from PipelineTS.spinesTS.base._torch_mixin import CQRWrapper, SinkhornResidualGate
