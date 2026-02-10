from PipelineTS.spinesTS.nn import DeepAR
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.spines_base import SpinesNNModelMixin


class DeepARModel(SpinesNNModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=30,
            d_model=64,
            n_blocks=3,
            n_rwkv_blocks=3,
            dropout=0.1,
            quantile=0.9,
            random_state=None,
            learning_rate=0.001,
            accelerator='auto',
            verbose=False,
            epochs=3000,
            batch_size='auto',
            patience=100,
            min_delta=0,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            loss_type='min'
    ):
        """
        DeepARModel: Probabilistic time series forecasting with autoregressive recurrent networks.

        Uses a modern RWKV (linear RNN) encoder with a Gaussian probabilistic output head.
        During training, the model learns distribution parameters (μ, σ) via Gaussian NLL loss.
        At inference, point predictions use the learned mean.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 30
            The number of lagged values to use as input features for training and prediction.
        d_model : int, optional, default: 64
            Hidden dimension size for the RWKV encoder and gated residual blocks.
        n_blocks : int, optional, default: 3
            Number of gated residual refinement blocks.
        n_rwkv_blocks : int, optional, default: 3
            Number of RWKV temporal mixing blocks in the encoder.
        dropout : float, optional, default: 0.1
            The dropout rate.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        learning_rate : float, optional, default: 0.001
            The learning rate for training.
        accelerator : str, optional, default: 'auto'
            The accelerator to use during training.
        verbose : bool, optional, default: False
            Whether to display verbose output during training.
        epochs : int, optional, default: 3000
            The number of epochs for training the model.
        batch_size : int or 'auto', optional, default: 'auto'
            The batch size used during training.
        patience : int, optional, default: 100
            The patience parameter for early stopping during training.
        min_delta : int, optional, default: 0
            The minimum change in the monitored quantity to qualify as an improvement.
        lr_scheduler : str, optional, default: 'CosineAnnealingLR'
            The learning rate scheduler used during training.
        lr_scheduler_patience : int, optional, default: 10
            The patience parameter for the learning rate scheduler.
        lr_factor : float, optional, default: 0.7
            The factor by which the learning rate is reduced.
        restore_best_weights : bool, optional, default: True
            Whether to restore the model weights from the best epoch.
        loss_type : str, optional, default: 'min'
            The type of loss used for training.

        Attributes
        ----------
        model : spinesTS.nn.DeepAR
            The DeepAR model from the spinesTS library.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            DeepAR,
            in_features=lags,
            out_features=lags,
            d_model=d_model,
            n_blocks=n_blocks,
            n_rwkv_blocks=n_rwkv_blocks,
            dropout=dropout,
            loss_fn='mae',
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
        )

        self.last_dt = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': (0, 0),
                'verbose': verbose,
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
                'min_delta': min_delta,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_patience': lr_scheduler_patience,
                'lr_factor': lr_factor,
                'restore_best_weights': restore_best_weights,
                'loss_type': loss_type
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the DeepAR model from the spinesTS library.

        Returns
        -------
        spinesTS.nn.DeepAR
            The DeepAR model.
        """
        return DeepAR(**self.all_configs['model_configs'])
