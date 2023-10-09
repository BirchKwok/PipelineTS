import logging

import torch
from darts.models.forecasting.tft_model import TFTModel as tft
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base import NNModelMixin, DartsForecastMixin, IntervalEstimationMixin

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
        )
        self.model = tft(**self.all_configs['model_configs'])

        self.all_configs.update(
            {
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'enable_progress_bar': enable_progress_bar,
                'enable_model_summary': enable_model_summary
            }
        )

    def fit(self, data, convert_dataframe_kwargs=None, cv=5, fit_kwargs=None):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs)
        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval(
                    data, estimator=tft, cv=cv, fit_kwargs=fit_kwargs
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
