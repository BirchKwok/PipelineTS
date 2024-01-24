import logging

import torch
from darts.models.forecasting.tide_model import TiDEModel as TiDE
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.base import NNModelMixin, IntervalEstimationMixin
from PipelineTS.base.darts_base import DartsForecastMixin
from PipelineTS.utils import update_dict_without_conflict

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class TiDEModel(DartsForecastMixin, NNModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            num_encoder_layers=1,
            num_decoder_layers=1,
            decoder_output_dim=16,
            hidden_size=128,
            temporal_width_past=4,
            temporal_width_future=4,
            temporal_decoder_hidden=32,
            use_layer_norm=False,
            dropout=0.1,
            use_static_covariates=True,
            loss_fn=torch.nn.MSELoss(),
            torch_metrics=None,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            lr_scheduler_cls=None,
            lr_scheduler_kwargs=None,
            use_reversible_instance_norm=False,
            batch_size=32,
            n_epochs=100,
            nr_epochs_val_period=1,
            add_encoders=None,
            enable_progress_bar=False,
            enable_model_summary=False,
            pl_trainer_kwargs=None,
            quantile=0.9,
            random_state=None,
            accelerator=None
    ):
        """
        TiDEModel: A wrapper for the TiDE model from the darts library with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        num_encoder_layers : int, optional, default: 1
            The number of encoder layers in the TiDE model.
        num_decoder_layers : int, optional, default: 1
            The number of decoder layers in the TiDE model.
        decoder_output_dim : int, optional, default: 16
            The output dimension of the decoder in the TiDE model.
        hidden_size : int, optional, default: 128
            The size of the hidden layer in the TiDE model.
        temporal_width_past : int, optional, default: 4
            The temporal width for past information in the TiDE model.
        temporal_width_future : int, optional, default: 4
            The temporal width for future information in the TiDE model.
        temporal_decoder_hidden : int, optional, default: 32
            The size of the hidden layer in the temporal decoder of the TiDE model.
        use_layer_norm : bool, optional, default: False
            Whether to use layer normalization in the TiDE model.
        dropout : float, optional, default: 0.1
            The dropout rate used during training.
        use_static_covariates : bool, optional, default: True
            Whether to use static covariates in the TiDE model.
        loss_fn : torch.nn.Module, optional, default: torch.nn.MSELoss()
            The loss function used for training the model.
        torch_metrics : list or None, optional, default: None
            The list of additional metrics to use during training. Set to None if no additional metrics are required.
        optimizer_cls : torch.optim.Optimizer, optional, default: torch.optim.Adam
            The optimizer used for training the model.
        optimizer_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the optimizer. Set to None if no additional arguments are required.
        lr_scheduler_cls : torch.optim.lr_scheduler._LRScheduler, optional, default: None
            The learning rate scheduler used during training. Set to None if no scheduler is required.
        lr_scheduler_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the learning rate scheduler. Set to None if no additional arguments are required.
        use_reversible_instance_norm : bool, optional, default: False
            Whether to use reversible instance normalization in the TiDE model.
        batch_size : int, optional, default: 32
            The batch size used during training.
        n_epochs : int, optional, default: 100
            The number of epochs for training the model.
        nr_epochs_val_period : int, optional, default: 1
            The frequency at which to perform validation during training.
        add_encoders : dict or None, optional, default: None
            The dictionary containing additional encoders for input features. Set to None if no additional encoders are required.
        enable_progress_bar : bool, optional, default: False
            Whether to display a progress bar during training.
        enable_model_summary : bool, optional, default: False
            Whether to display a summary of the model architecture.
        pl_trainer_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the PyTorch Lightning trainer. Set to None if no additional arguments are required.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        accelerator : str or None, optional, default: None
            The PyTorch Lightning accelerator to use during training. Set to None for the default accelerator.

        Attributes
        ----------
        model : darts.models.forecasting.tide_model.TiDEModel
            The TiDE model from the darts library.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        if pl_trainer_kwargs is None:
            pl_trainer_kwargs = {}

        pl_trainer_kwargs = update_dict_without_conflict(pl_trainer_kwargs, {
            'accelerator': self.accelerator,
            'enable_progress_bar': enable_progress_bar,
            'enable_model_summary': enable_model_summary
        })

        self.all_configs['model_configs'] = generate_function_kwargs(
            TiDE,
            input_chunk_length=lags,
            output_chunk_length=lags,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_width_past=temporal_width_past,
            temporal_width_future=temporal_width_future,
            temporal_decoder_hidden=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            use_static_covariates=use_static_covariates,
            loss_fn=loss_fn,
            torch_metrics=torch_metrics,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            use_reversible_instance_norm=use_reversible_instance_norm,
            batch_size=batch_size,
            n_epochs=n_epochs,
            nr_epochs_val_period=nr_epochs_val_period,
            add_encoders=add_encoders,
            pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=random_state,
        )
        self.model = self._define_model()

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'enable_progress_bar': enable_progress_bar,
                'enable_model_summary': enable_model_summary
            }
        )

    def _define_model(self):
        """
        Define the TiDE model from the darts library.

        Returns
        -------
        darts.models.forecasting.tide_model.TiDEModel
            The TiDE model from the darts library.
        """
        return TiDE(**self.all_configs['model_configs'])
