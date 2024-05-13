from PipelineTS.spinesTS.nn import Time2VecNet
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.spines_base import SpinesNNModelMixin


class Time2VecModel(SpinesNNModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=30,
            quantile=0.9,
            random_state=None,
            learning_rate=0.01,
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
        Time2VecModel: A wrapper for the Time2VecNet model with additional features.

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
        learning_rate : float, optional, default: 0.01
            The learning rate for training the model.
        accelerator : str, optional, default: 'auto'
            The PyTorch Lightning accelerator to use during training.
        verbose : bool, optional, default: False
            Whether to display detailed information during training.
        epochs : int, optional, default: 1000
            The number of epochs for training the model.
        batch_size : int or 'auto', optional, default: 'auto'
            The batch size used during training. Set to 'auto' for automatic batch size determination.
        patience : int, optional, default: 100
            The patience parameter for early stopping during training.
        min_delta : float, optional, default: 0
            The minimum change in the monitored quantity to qualify as an improvement during training.
        lr_scheduler : str, optional, default: 'CosineAnnealingLR'
            The learning rate scheduler to use during training.
        lr_scheduler_patience : int, optional, default: 10
            The patience parameter for the learning rate scheduler.
        lr_factor : float, optional, default: 0.7
            The factor by which the learning rate is reduced during training.
        restore_best_weights : bool, optional, default: True
            Whether to restore the best weights during training when using early stopping.
        loss_type : str, optional, default: 'min'
            The type of loss function to use. Supported values include 'min' and 'max'.

        Attributes
        ----------
        x : None
            Placeholder for input data (not used in this implementation).
        model : spinesTS.nn.Time2VecNet
            The Time2VecNet model from the spinesTS library.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            Time2VecNet,
            in_features=lags,
            out_features=lags,
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
        Define the Time2VecNet model from the spinesTS library.

        Returns
        -------
        spinesTS.nn.Time2VecNet
            The Time2VecNet model from the spinesTS library.
        """
        return Time2VecNet(**self.all_configs['model_configs'])
