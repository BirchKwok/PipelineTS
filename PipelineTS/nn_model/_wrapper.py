from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.model_mixins import NNForecastingMixin, MultivariateNNForecastingMixin


class NNBackboneForecastingMixin(NNForecastingMixin):
    backbone_cls = None

    def _init_backbone_model(
            self,
            backbone_cls,
            lags,
            quantile,
            time_col,
            target_col,
            verbose,
            epochs,
            batch_size,
            patience,
            min_delta,
            lr_scheduler,
            lr_scheduler_patience,
            lr_factor,
            restore_best_weights,
            loss_type,
            use_ema,
            ema_decay,
            use_swa,
            swa_start_frac,
            warmup_epochs,
            use_residual_gate,
            model_kwargs,
            extra_configs=None,
    ):
        self.backbone_cls = backbone_cls
        self.all_configs['model_configs'] = generate_function_kwargs(
            backbone_cls,
            in_features=lags,
            out_features=lags,
            **model_kwargs,
        )
        self.last_dt = None
        configs = {
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
            'verbose': verbose,
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'min_delta': min_delta,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_patience': lr_scheduler_patience,
            'lr_factor': lr_factor,
            'restore_best_weights': restore_best_weights,
            'loss_type': loss_type,
            'use_ema': use_ema,
            'ema_decay': ema_decay,
            'use_swa': use_swa,
            'swa_start_frac': swa_start_frac,
            'warmup_epochs': warmup_epochs,
            'use_residual_gate': use_residual_gate,
        }
        if extra_configs:
            configs.update(extra_configs)
        self.all_configs.update(configs)
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return self.backbone_cls(**self.all_configs['model_configs'])


class MultivariateNNBackboneForecastingMixin(MultivariateNNForecastingMixin):
    backbone_cls = None

    def _init_backbone_model(
            self,
            backbone_cls,
            lags,
            quantile,
            time_col,
            verbose,
            epochs,
            batch_size,
            patience,
            min_delta,
            lr_scheduler,
            lr_scheduler_patience,
            lr_factor,
            restore_best_weights,
            loss_type,
            use_ema,
            ema_decay,
            use_swa,
            swa_start_frac,
            warmup_epochs,
            use_residual_gate,
            model_kwargs,
            target_col=None,
            extra_configs=None,
    ):
        self.backbone_cls = backbone_cls
        self.all_configs['model_configs'] = generate_function_kwargs(
            backbone_cls,
            in_features=lags,
            out_features=lags,
            **model_kwargs,
        )
        self.last_dt = None
        configs = {
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': self._primary_target if target_col is None else target_col,
            'quantile_error': (0, 0),
            'verbose': verbose,
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'min_delta': min_delta,
            'lr_scheduler': lr_scheduler,
            'lr_scheduler_patience': lr_scheduler_patience,
            'lr_factor': lr_factor,
            'restore_best_weights': restore_best_weights,
            'loss_type': loss_type,
            'use_ema': use_ema,
            'ema_decay': ema_decay,
            'use_swa': use_swa,
            'swa_start_frac': swa_start_frac,
            'warmup_epochs': warmup_epochs,
            'use_residual_gate': use_residual_gate,
        }
        if extra_configs:
            configs.update(extra_configs)
        self.all_configs.update(configs)
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return self.backbone_cls(**self.all_configs['model_configs'])
