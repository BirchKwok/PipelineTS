from PipelineTS.spinesTS.nn import GAUNet
from spinesUtils.asserts import generate_function_kwargs
from PipelineTS.base.spines_base import SpinesNNModelMixin


class GAUModel(SpinesNNModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=30,
            quantile=0.9,
            random_state=None,
            level=2,
            learning_rate=0.001,
            accelerator='auto',
            verbose=False,
            epochs=1000,
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
        GAUModel: A wrapper for the GAUNet neural network model with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 30
            The number of lagged values to use as input features for training and prediction.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        level : int, optional, default: 2
            The level parameter for the GAUNet model.
        learning_rate : float, optional, default: 0.001
            The learning rate for training the GAUNet model.
        accelerator : str, optional, default: 'auto'
            The PyTorch Lightning accelerator to use during training.
        verbose : bool, optional, default: False
            Whether to print detailed information during training.
        epochs : int, optional, default: 1000
            The number of epochs for training the GAUNet model.
        batch_size : int or 'auto', optional, default: 'auto'
            The batch size used during training. If set to 'auto', it is determined automatically.
        patience : int, optional, default: 100
            The patience parameter for early stopping during training.
        min_delta : float, optional, default: 0
            The minimum change in the monitored quantity to qualify as an improvement during early stopping.
        lr_scheduler : str, optional, default: 'CosineAnnealingLR'
            The learning rate scheduler used during training.
        lr_scheduler_patience : int, optional, default: 10
            The patience parameter for the learning rate scheduler.
        lr_factor : float, optional, default: 0.7
            The factor by which the learning rate is reduced during training.
        restore_best_weights : bool, optional, default: True
            Whether to restore the model weights from the epoch with the best value of the monitored quantity.
        loss_type : str, optional, default: 'min'
            The type of loss function to use during training.

        Attributes
        ----------
        model : spinesTS.nn.GAUNet
            The GAUNet neural network model.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            GAUNet,
            in_features=lags,
            out_features=lags,
            level=level,
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
            loss_fn='mae'
        )

        self.last_dt = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0,
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
        Define the GAUNet neural network model.

        Returns
        -------
        spinesTS.nn.GAUNet
            The GAUNet neural network model.
        """
        return GAUNet(**self.all_configs['model_configs'])
