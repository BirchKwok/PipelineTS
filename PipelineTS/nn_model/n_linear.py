import torch
from torch import nn

from darts.models.forecasting.nlinear import NLinearModel as n_linear
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base import NNModelMixin, DartsForecastMixin, IntervalEstimationMixin


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
            loss_fn=nn.MSELoss(),
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
            pl_trainer_kwargs=None,
            quantile=0.9,
            random_state=None,
    ):
        super().__init__()

        if pl_trainer_kwargs is not None and 'accelerator' not in pl_trainer_kwargs:
            pl_trainer_kwargs.update({'accelerator': self.device})
        elif pl_trainer_kwargs is None:
            pl_trainer_kwargs = {'accelerator': self.device}

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
        self.model = n_linear(**self.all_configs['model_configs'])

        self.all_configs.update(
            {
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
            }
        )

    def fit(self, data, convert_dataframe_kwargs=None, cv=5, fit_kwargs=None):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs)

        self.all_configs['lower_limit'], self.all_configs['higher_limit'] = \
            self.calculate_confidence_interval(
                data, estimator=n_linear, cv=cv, fit_kwargs=fit_kwargs
            )

        return self

    def predict(self, n, **kwargs):
        res = super().predict(n, predict_likelihood_parameters=False, **kwargs)

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.rename_prediction(res)
