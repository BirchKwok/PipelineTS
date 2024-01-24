import logging, platform

import torch
from darts.models.forecasting.tft_model import TFTModel as tft
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.base import NNModelMixin, IntervalEstimationMixin
from PipelineTS.base.darts_base import DartsForecastMixin
from PipelineTS.utils import update_dict_without_conflict

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class TFTModel(DartsForecastMixin, NNModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            hidden_size=16,
            lstm_layers=1,
            num_attention_heads=4,
            full_attention=False,
            feed_forward='GatedResidualNetwork',
            dropout=0.1,
            hidden_continuous_size=8,
            categorical_embedding_sizes=None,
            add_relative_index=True,
            norm_type='LayerNorm',
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
        TFTModel: A wrapper for the TFT model from the darts library with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        hidden_size : int, optional, default: 16
            The size of the hidden layer in the TFT model.
        lstm_layers : int, optional, default: 1
            The number of LSTM layers in the TFT model.
        num_attention_heads : int, optional, default: 4
            The number of attention heads in the TFT model.
        full_attention : bool, optional, default: False
            Whether to use full attention in the TFT model.
        feed_forward : str, optional, default: 'GatedResidualNetwork'
            The type of feed-forward layer used in the TFT model.
        dropout : float, optional, default: 0.1
            The dropout rate used during training.
        hidden_continuous_size : int, optional, default: 8
            The size of the hidden continuous layer in the TFT model.
        categorical_embedding_sizes : dict or None, optional, default: None
            The sizes of the embedding layers for categorical variables. Set to None if no categorical variables are present.
        add_relative_index : bool, optional, default: True
            Whether to add a relative index to the input features in the TFT model.
        norm_type : str, optional, default: 'LayerNorm'
            The type of normalization layer used in the TFT model.
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
            Whether to use reversible instance normalization in the TFT model.
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
        model : darts.models.forecasting.tft_model.TFTModel
            The TFT model from the darts library.
        """
        if platform.system() == 'Darwin' and torch.backends.mps.is_available():
            # Since using mps backend for tft model on Darwin system gives an error:
            # "The operator 'aten::upsample_linear1d.out' is not currently implemented for the MPS device",
            # the cpu backend is used as an alternative
            accelerator = 'cpu'

        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        if pl_trainer_kwargs is None:
            pl_trainer_kwargs = {}

        pl_trainer_kwargs = update_dict_without_conflict(pl_trainer_kwargs, {
            'accelerator': self.accelerator,
            'enable_progress_bar': enable_progress_bar,
            'enable_model_summary': enable_model_summary,
        })

        self.all_configs['model_configs'] = generate_function_kwargs(
            tft,
            input_chunk_length=lags,
            output_chunk_length=lags,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            full_attention=full_attention,
            feed_forward=feed_forward,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            categorical_embedding_sizes=categorical_embedding_sizes,
            add_relative_index=add_relative_index,
            loss_fn=loss_fn,
            norm_type=norm_type,
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
            likelihood=None,
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
        Define the TFT model from the darts library.

        Returns
        -------
        darts.models.forecasting.tft_model.TFTModel
            The TFT model from the darts library.
        """
        return tft(**self.all_configs['model_configs'])
