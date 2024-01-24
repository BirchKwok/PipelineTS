import logging

import torch
from darts.models.forecasting.transformer_model import TransformerModel as tfm
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.base import NNModelMixin, IntervalEstimationMixin
from PipelineTS.base.darts_base import DartsForecastMixin
from PipelineTS.utils import update_dict_without_conflict

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class TransformerModel(DartsForecastMixin, NNModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            activation='relu',
            norm_type=None,
            custom_encoder=None,
            custom_decoder=None,
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
        TransformerModel: A wrapper for the TransformerModel from the darts library with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        d_model : int, optional, default: 64
            Dimensionality of the model.
        nhead : int, optional, default: 4
            Number of heads in the multiheadattention models.
        num_encoder_layers : int, optional, default: 3
            Number of sub-encoder-layers in the encoder.
        num_decoder_layers : int, optional, default: 3
            Number of sub-decoder-layers in the decoder.
        dim_feedforward : int, optional, default: 512
            Dimension of the feedforward network model.
        dropout : float, optional, default: 0.1
            The dropout value for training.
        activation : str, optional, default: 'relu'
            The activation function to use. Default is 'relu'.
        norm_type : str, optional, default: None
            Normalization type to apply on the input data. Default is None.
        custom_encoder : None, optional, default: None
            Custom encoder module to replace the default transformer encoder.
        custom_decoder : None, optional, default: None
            Custom decoder module to replace the default transformer decoder.
        loss_fn : torch.nn.modules.loss._Loss, optional, default: torch.nn.MSELoss()
            Loss function for training the model.
        torch_metrics : None, optional, default: None
            Additional metrics to track during training.
        optimizer_cls : torch.optim.Optimizer, optional, default: torch.optim.Adam
            Optimizer class for training the model.
        optimizer_kwargs : None, optional, default: None
            Additional keyword arguments for the optimizer.
        lr_scheduler_cls : None, optional, default: None
            Learning rate scheduler class for adjusting the learning rate during training.
        lr_scheduler_kwargs : None, optional, default: None
            Additional keyword arguments for the learning rate scheduler.
        use_reversible_instance_norm : bool, optional, default: False
            Whether to use reversible instance normalization.
        batch_size : int, optional, default: 32
            Batch size used during training.
        n_epochs : int, optional, default: 100
            Number of epochs for training.
        nr_epochs_val_period : int, optional, default: 1
            Number of epochs per validation period.
        add_encoders : None, optional, default: None
            Additional encoders to concatenate with the input data.
        enable_progress_bar : bool, optional, default: False
            Whether to display a progress bar during training.
        enable_model_summary : bool, optional, default: False
            Whether to display the model summary.
        pl_trainer_kwargs : None, optional, default: None
            Additional keyword arguments for the PyTorch Lightning trainer.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        accelerator : str, optional, default: None
            The PyTorch Lightning accelerator to use during training.

        Attributes
        ----------
        x : None
            Placeholder for input data (not used in this implementation).
        model : darts.models.forecasting.transformer_model.TransformerModel
            The TransformerModel model from the darts library.
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
            tfm,
            input_chunk_length=lags,
            output_chunk_length=lags,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
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
        Define the TransformerModel model from the darts library.

        Returns
        -------
        darts.models.forecasting.transformer_model.TransformerModel
            The TransformerModel model from the darts library.
        """
        return tfm(**self.all_configs['model_configs'])
