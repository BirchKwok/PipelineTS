import logging
import platform

import torch
from darts.models.forecasting.nhits import NHiTSModel as n_hits
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.base import NNModelMixin, IntervalEstimationMixin
from PipelineTS.base.darts_base import DartsForecastMixin
from PipelineTS.utils import update_dict_without_conflict

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class NHitsModel(DartsForecastMixin, NNModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            num_stacks=3,
            num_blocks=1,
            num_layers=2,
            layer_widths=512,
            pooling_kernel_sizes=None,
            n_freq_downsample=None,
            dropout=0.1,
            activation='ReLU',
            MaxPool1d=True,
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
        NHitsModel: A wrapper for the NHiTSModel from the darts library with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        num_stacks : int, optional, default: 3
            The number of stacks in the architecture.
        num_blocks : int, optional, default: 1
            The number of blocks in each stack.
        num_layers : int, optional, default: 2
            The number of fully connected layers in each block.
        layer_widths : int or list, optional, default: 512
            The number of neurons in each fully connected layer. Can be an integer or a list for variable widths.
        pooling_kernel_sizes : list or None, optional, default: None
            The kernel sizes for pooling layers. If None, no pooling is applied.
        n_freq_downsample : int or None, optional, default: None
            The factor by which to downsample the frequency domain. If None, no downsampling is applied.
        dropout : float, optional, default: 0.1
            The dropout rate applied to each fully connected layer.
        activation : str, optional, default: 'ReLU'
            The activation function used in the fully connected layers.
        MaxPool1d : bool, optional, default: True
            Whether to use max pooling in the architecture.
        loss_fn : torch.nn.Module, optional, default: torch.nn.MSELoss()
            The loss function used for training the model.
        torch_metrics : list or None, optional, default: None
            Additional metrics to track during training.
        optimizer_cls : torch.optim.Optimizer, optional, default: torch.optim.Adam
            The optimizer class used for training.
        optimizer_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the optimizer.
        lr_scheduler_cls : torch.optim.lr_scheduler._LRScheduler or None, optional, default: None
            The learning rate scheduler class used for training.
        lr_scheduler_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the learning rate scheduler.
        use_reversible_instance_norm : bool, optional, default: False
            Whether to use reversible instance normalization.
        batch_size : int, optional, default: 32
            The batch size used during training.
        n_epochs : int, optional, default: 100
            The number of epochs for training the model.
        nr_epochs_val_period : int, optional, default: 1
            The period for validating the model during training.
        add_encoders : dict or None, optional, default: None
            Additional encoder configurations.
        enable_progress_bar : bool, optional, default: False
            Whether to display a progress bar during training.
        enable_model_summary : bool, optional, default: False
            Whether to print the model summary.
        pl_trainer_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the PyTorch Lightning trainer.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        accelerator : str or None, optional, default: None
            The PyTorch Lightning accelerator to use during training.

        Attributes
        ----------
        model : darts.models.forecasting.nhits.NHiTSModel
            The NHiTSModel from the darts library.
        """
        if platform.system() == 'Darwin' and torch.backends.mps.is_available():
            # Since using mps backend for n_hits model on Darwin system gives an error:
            # "The operator 'aten::upsample_linear1d.out' is not currently implemented for the MPS device",
            # the cpu backend is used as an alternative
            accelerator = 'cpu'

        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        if pl_trainer_kwargs is None:
            pl_trainer_kwargs = {}

        pl_trainer_kwargs = update_dict_without_conflict(pl_trainer_kwargs, {
            'accelerator': self.accelerator,
            'enable_progress_bar': enable_progress_bar,
            'enable_model_summary': enable_model_summary
        })

        self.all_configs['model_configs'] = generate_function_kwargs(
            n_hits,
            input_chunk_length=lags,
            output_chunk_length=lags,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            pooling_kernel_sizes=pooling_kernel_sizes,
            n_freq_downsample=n_freq_downsample,
            dropout=dropout,
            activation=activation,
            MaxPool1d=MaxPool1d,
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
        Define the NHiTSModel from the darts library.

        Returns
        -------
        darts.models.forecasting.nhits.NHiTSModel
            The NHiTSModel from the darts library.
        """
        return n_hits(**self.all_configs['model_configs'])
