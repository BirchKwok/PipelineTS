from PipelineTS.models.nn import backbones as _backbones
from PipelineTS.models.nn._modern_ts_specs import MODERN_TS_MODEL_SPECS
from PipelineTS.models.nn._wrapper import NNBackboneForecastingMixin


class _ModernTSForecastingModel(NNBackboneForecastingMixin):
    backbone_cls = None

    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            d_model=64,
            n_heads=4,
            nhead=None,
            e_layers=2,
            num_encoder_layers=None,
            d_ff=128,
            dim_feedforward=None,
            patch_len=4,
            stride=None,
            dropout=0.1,
            use_revin=True,
            moving_avg=3,
            top_k=3,
            quantile=0.9,
            random_state=None,
            learning_rate=0.001,
            accelerator='auto',
            verbose=False,
            epochs=1500,
            batch_size='auto',
            patience=80,
            min_delta=0,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            loss_type='min',
            weight_decay=1e-4,
            use_gtb=False,
            gtb_d_model=64,
            routing_mode='static',
            use_ema=False,
            ema_decay=0.999,
            use_swa=False,
            swa_start_frac=0.75,
            warmup_epochs=0,
            use_residual_gate=False,
            **backbone_kwargs,
    ):
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self._init_backbone_model(
            backbone_cls=self.backbone_cls,
            lags=lags,
            quantile=quantile,
            time_col=time_col,
            target_col=target_col,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            lr_scheduler=lr_scheduler,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            restore_best_weights=restore_best_weights,
            loss_type=loss_type,
            use_ema=use_ema,
            ema_decay=ema_decay,
            use_swa=use_swa,
            swa_start_frac=swa_start_frac,
            warmup_epochs=warmup_epochs,
            use_residual_gate=use_residual_gate,
            model_kwargs=dict(
            d_model=d_model,
            n_heads=n_heads if nhead is None else nhead,
            e_layers=e_layers if num_encoder_layers is None else num_encoder_layers,
            d_ff=d_ff if dim_feedforward is None else dim_feedforward,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
            use_revin=use_revin,
            moving_avg=moving_avg,
            top_k=top_k,
            loss_fn='huber',
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
            weight_decay=weight_decay,
            use_gtb=use_gtb,
            gtb_d_model=gtb_d_model,
            routing_mode=routing_mode,
            **backbone_kwargs,
            ),
        )


def _build_forecasting_model_class(class_name, backbone_cls):
    return type(class_name, (_ModernTSForecastingModel,), {'backbone_cls': backbone_cls, '__module__': __name__})


MODERN_TS_MODEL_CLASSES = {}

for _spec in MODERN_TS_MODEL_SPECS:
    _cls = _build_forecasting_model_class(_spec.wrapper_class, getattr(_backbones, _spec.backbone_class))
    globals()[_spec.wrapper_class] = _cls
    MODERN_TS_MODEL_CLASSES[_spec.key] = _cls


__all__ = ['MODERN_TS_MODEL_CLASSES'] + [spec.wrapper_class for spec in MODERN_TS_MODEL_SPECS]
