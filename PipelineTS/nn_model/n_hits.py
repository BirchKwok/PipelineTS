import logging

import torch
from darts.models.forecasting.nhits import NHiTSModel as n_hits
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base import NNModelMixin, DartsForecastMixin, IntervalEstimationMixin

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
        return n_hits(**self.all_configs['model_configs'])

    def fit(self, data, cv=5, convert_dataframe_kwargs=None, fit_kwargs=None, convert_float32=True):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs, convert_float32=convert_float32)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(
                    data, fit_kwargs=fit_kwargs, convert2dts_dataframe_kwargs=convert_dataframe_kwargs,
                    cv=cv
                )

        return self

    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        res = super().predict(n, predict_likelihood_parameters=False, **predict_kwargs)
        res = self.rename_prediction(res)
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
