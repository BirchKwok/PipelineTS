import logging
import torch
from darts.models.forecasting.dlinear import DLinearModel as d_linear
from spinesUtils.asserts import generate_function_kwargs
from PipelineTS.base.base import NNModelMixin, IntervalEstimationMixin
from PipelineTS.base.darts_base import DartsForecastMixin
from PipelineTS.utils import update_dict_without_conflict

# Suppressing unnecessary PyTorch Lightning warnings
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class DLinearModel(DartsForecastMixin, NNModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            shared_weights=False,
            kernel_size=25,
            const_init=True,
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
        DLinearModel: A wrapper for Darts' DLinear forecasting model with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        shared_weights : bool, optional, default: False
            Whether to share weights across all inputs in the DLinear model.
        kernel_size : int, optional, default: 25
            The size of the kernel in the DLinear model.
        const_init : bool, optional, default: True
            Whether to use constant initialization for weights in the DLinear model.
        use_static_covariates : bool, optional, default: True
            Whether to use static covariates in the DLinear model.
        loss_fn : torch.nn.Module, optional, default: torch.nn.MSELoss
            The loss function used during training.
        torch_metrics : torch.nn.Module or None, optional, default: None
            Additional metrics for training evaluation.
        optimizer_cls : torch.optim.Optimizer, optional, default: torch.optim.Adam
            The optimizer used for training.
        optimizer_kwargs : dict or None, optional, default: None
            Additional keyword arguments for configuring the optimizer.
        lr_scheduler_cls : torch.optim.lr_scheduler, optional, default: None
            The learning rate scheduler used during training.
        lr_scheduler_kwargs : dict or None, optional, default: None
            Additional keyword arguments for configuring the learning rate scheduler.
        use_reversible_instance_norm : bool, optional, default: False
            Whether to use reversible instance normalization in the DLinear model.
        batch_size : int, optional, default: 32
            The batch size used during training.
        n_epochs : int, optional, default: 100
            The number of epochs for training.
        nr_epochs_val_period : int, optional, default: 1
            The number of epochs between each validation period during training.
        add_encoders : list or None, optional, default: None
            Additional encoders for the DLinear model.
        enable_progress_bar : bool, optional, default: False
            Whether to enable the progress bar during training.
        enable_model_summary : bool, optional, default: False
            Whether to enable the model summary during training.
        pl_trainer_kwargs : dict or None, optional, default: None
            Additional keyword arguments for configuring the PyTorch Lightning trainer.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        accelerator : str or None, optional, default: None
            The PyTorch Lightning accelerator to use during training.

        Attributes
        ----------
        model : darts.models.forecasting.dlinear.DLinearModel
            The Darts DLinear forecasting model.
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
            d_linear,
            input_chunk_length=lags,
            output_chunk_length=lags,
            shared_weights=shared_weights,
            kernel_size=kernel_size,
            const_init=const_init,
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
        Define the DLinear forecasting model.

        Returns
        -------
        darts.models.forecasting.dlinear.DLinearModel
            The Darts DLinear forecasting model.
        """
        return d_linear(**self.all_configs['model_configs'])
