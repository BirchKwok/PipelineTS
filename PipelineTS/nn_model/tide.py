import logging

import pandas as pd
import torch
from darts.models.forecasting.tide_model import TiDEModel as TiDE
from spinesUtils import ParameterTypeAssert
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base import NNModelMixin, DartsForecastMixin, IntervalEstimationMixin

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
        super().__init__(time_col=time_col, target_col=target_col, device=accelerator)

        if pl_trainer_kwargs is not None and 'accelerator' not in pl_trainer_kwargs:
            pl_trainer_kwargs.update({'accelerator': self.device})
        elif pl_trainer_kwargs is None:
            pl_trainer_kwargs = {'accelerator': self.device}

        if 'enable_progress_bar' not in pl_trainer_kwargs:
            pl_trainer_kwargs.update({'enable_progress_bar': enable_progress_bar})
        if 'enable_model_summary' not in pl_trainer_kwargs:
            pl_trainer_kwargs.update({'enable_model_summary': enable_model_summary})

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
        return TiDE(**self.all_configs['model_configs'])

    def fit(self, data, cv=5, convert_dataframe_kwargs=None, fit_kwargs=None):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs, convert_float32=True)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(
                    data, fit_kwargs=fit_kwargs, convert2dts_dataframe_kwargs=convert_dataframe_kwargs,
                    cv=cv
                )

        return self

    @ParameterTypeAssert({
        'n': int,
        'series': (pd.DataFrame, None),
        'predict_kwargs': (None, dict),
        'convert_dataframe_kwargs': (None, dict),
    })
    def predict(self, n, series=None, predict_kwargs=None, convert_dataframe_kwargs=None, convert_float32=True):
        if predict_kwargs is None:
            predict_kwargs = {}

        res = super().predict(n, series=series,
                              predict_kwargs=predict_kwargs, convert_dataframe_kwargs=convert_dataframe_kwargs,
                              convert_float32=True)
        res = self.rename_prediction(res)
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
