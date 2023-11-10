import logging

import pandas as pd
import torch
from darts.models.forecasting.nlinear import NLinearModel as n_linear
from spinesUtils import ParameterTypeAssert
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base import NNModelMixin, DartsForecastMixin, IntervalEstimationMixin
from PipelineTS.utils import update_dict_without_conflict

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


class NLinearModel(DartsForecastMixin, NNModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            shared_weights=False,
            const_init=True,
            normalize=False,
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
        NLinearModel: A wrapper for the NLinearModel from the darts library with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        shared_weights : bool, optional, default: False
            Whether to use shared weights across different lags.
        const_init : bool, optional, default: True
            Whether to initialize the weights with constants.
        normalize : bool, optional, default: False
            Whether to normalize the input data.
        use_static_covariates : bool, optional, default: True
            Whether to include static covariates in the model.
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
        model : darts.models.forecasting.nlinear.NLinearModel
            The NLinearModel from the darts library.
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
            n_linear,
            input_chunk_length=lags,
            output_chunk_length=lags,
            shared_weights=shared_weights,
            const_init=const_init,
            normalize=normalize,
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
        Define the NLinearModel from the darts library.

        Returns
        -------
        darts.models.forecasting.nlinear.NLinearModel
            The NLinearModel from the darts library.
        """
        return n_linear(**self.all_configs['model_configs'])

    def fit(self, data, cv=5, convert_dataframe_kwargs=None, fit_kwargs=None):
        """
        Train the NLinear model on the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing time and target columns.
        cv : int, optional, default: 5
            The number of cross-validation folds.
        convert_dataframe_kwargs : dict or None, optional, default: None
            Additional keyword arguments for converting the data to PyTorch tensors.
        fit_kwargs : dict or None, optional, default: None
            Additional keyword arguments for training the model.

        Returns
        -------
        self
            The fitted NLinearModel instance.
        """
        super().fit(data, convert_dataframe_kwargs, fit_kwargs, convert_float32=True)
        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(
                    data, fit_kwargs=fit_kwargs, convert2dts_dataframe_kwargs=convert_dataframe_kwargs,
                    cv=5
                )

        return self

    @ParameterTypeAssert({
        'n': int,
        'data': (pd.DataFrame, None),
        'predict_kwargs': (None, dict),
        'convert_dataframe_kwargs': (None, dict),
    })
    def predict(self, n, data=None, predict_kwargs=None, convert_dataframe_kwargs=None):
        """
        Make predictions using the fitted NLinear model.

        Parameters
        ----------
        n : int
            The number of time steps to predict.
        data : pd.DataFrame or None, optional, default: None
            The input data for prediction. If None, the last available data in the model is used.
        predict_kwargs : dict or None, optional, default: None
            Additional keyword arguments for making predictions.
        convert_dataframe_kwargs : dict or None, optional, default: None
            Additional keyword arguments for converting the data to PyTorch tensors.

        Returns
        -------
        pd.DataFrame
            The predicted values along with time information.
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        res = super().predict(n, data=data,
                              predict_kwargs=predict_kwargs, convert_dataframe_kwargs=convert_dataframe_kwargs,
                              convert_float32=True)
        res = self.rename_prediction(res)
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
