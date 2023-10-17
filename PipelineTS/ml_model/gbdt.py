import pandas as pd
from darts.models import (
    CatBoostModel as CBT,
    LightGBMModel as LGB,
    XGBModel as XGB,
    RandomForest as RF
)
from spinesUtils.asserts import generate_function_kwargs, ParameterTypeAssert

from PipelineTS.base import DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin


class CatBoostModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'lags_past_covariates': (None, int),
        'lags_future_covariates': (None, int),
        'quantile': (None, float),
        'random_state': (None, int),
        'multi_models': bool,
        'use_static_covariates': bool,
        'verbose': bool
    }, 'CatBoostModel')
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            lags_past_covariates=None,
            lags_future_covariates=None,
            add_encoders=None,
            quantile=0.9,
            random_state=None,
            multi_models=True,
            use_static_covariates=True,
            verbose=False,
            **darts_catboost_model_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            CBT,
            lags=lags+1,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=lags,
            add_encoders=add_encoders,
            random_state=random_state,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            verbose=verbose,
            **darts_catboost_model_configs
        )
        self.model = self._define_model()

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0
            }
        )

    def _define_model(self):
        return CBT(**self.all_configs['model_configs'])

    @ParameterTypeAssert({
        'data': pd.DataFrame,
        'convert_dataframe_kwargs': (None, dict),
        'cv': int,
        'fit_kwargs': (None, dict)
    })
    def fit(self, data, cv=5, convert_dataframe_kwargs=None, fit_kwargs=None):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self

    @ParameterTypeAssert({
        'n': int,
        'predict_kwargs': (None, dict)
    })
    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        res = self.model.predict(n, **predict_kwargs).pd_dataframe()
        res = self.rename_prediction(res)
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)


class LightGBMModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'lags_past_covariates': (None, int),
        'lags_future_covariates': (None, int),
        'quantile': (None, float),
        'random_state': (None, int),
        'multi_models': bool,
        'use_static_covariates': bool,
        'verbose': int
    }, 'LightGBMModel')
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            lags_past_covariates=None,
            lags_future_covariates=None,
            add_encoders=None,
            quantile=0.9,
            random_state=None,
            multi_models=True,
            use_static_covariates=True,
            categorical_past_covariates=None,
            categorical_future_covariates=None,
            categorical_static_covariates=None,
            verbose=-1,
            linear_tree=True,
            **darts_lightgbm_model_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            LGB,
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=lags,
            add_encoders=add_encoders,
            random_state=random_state,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            categorical_past_covariates=categorical_past_covariates,
            categorical_future_covariates=categorical_future_covariates,
            categorical_static_covariates=categorical_static_covariates,
            verbose=verbose,
            linear_tree=linear_tree,
            **darts_lightgbm_model_configs
        )
        self.model = self._define_model()

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0
            }
        )

    def _define_model(self):
        return LGB(**self.all_configs['model_configs'])

    @ParameterTypeAssert({
        'data': pd.DataFrame,
        'convert_dataframe_kwargs': (None, dict),
        'cv': int,
        'fit_kwargs': (None, dict)
    })
    def fit(self, data, convert_dataframe_kwargs=None, cv=5, fit_kwargs=None):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self

    @ParameterTypeAssert({
        'n': int,
        'predict_kwargs': (None, dict)
    })
    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        res = self.model.predict(n, **predict_kwargs).pd_dataframe()
        res = self.rename_prediction(res)
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)


class XGBoostModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            lags_past_covariates=None,
            lags_future_covariates=None,
            add_encoders=None,
            quantile=0.9,
            random_state=None,
            multi_models=True,
            use_static_covariates=True,
            categorical_past_covariates=None,
            categorical_future_covariates=None,
            categorical_static_covariates=None,
            verbose=0,
            **darts_xgboost_model_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            XGB,
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=lags,
            add_encoders=add_encoders,
            random_state=random_state,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            categorical_past_covariates=categorical_past_covariates,
            categorical_future_covariates=categorical_future_covariates,
            categorical_static_covariates=categorical_static_covariates,
            verbosity=verbose,
            **darts_xgboost_model_configs
        )
        self.model = self._define_model()

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0
            }
        )

    def _define_model(self):
        return XGB(**self.all_configs['model_configs'])

    @ParameterTypeAssert({
        'data': pd.DataFrame,
        'convert_dataframe_kwargs': (None, dict),
        'cv': int,
        'fit_kwargs': (None, dict)
    })
    def fit(self, data, convert_dataframe_kwargs=None, cv=5, fit_kwargs=None):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self

    @ParameterTypeAssert({
        'n': int,
        'predict_kwargs': (None, dict)
    })
    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        res = self.model.predict(n, **predict_kwargs).pd_dataframe()
        res = self.rename_prediction(res)
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)


class RandomForestModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            lags_past_covariates=None,
            lags_future_covariates=None,
            add_encoders=None,
            n_estimators=100,
            quantile=0.9,
            random_state=None,
            multi_models=True,
            use_static_covariates=True,
            **darts_random_forest_model_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            RF,
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=lags,
            add_encoders=add_encoders,
            n_estimators=n_estimators,
            random_state=random_state,
            multi_models=multi_models,
            use_static_covariates=use_static_covariates,
            **darts_random_forest_model_configs
        )
        self.model = self._define_model()

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0
            }
        )

    def _define_model(self):
        return RF(**self.all_configs['model_configs'])

    @ParameterTypeAssert({
        'data': pd.DataFrame,
        'convert_dataframe_kwargs': (None, dict),
        'cv': int,
        'fit_kwargs': (None, dict)
    })
    def fit(self, data, cv=5, convert_dataframe_kwargs=None, fit_kwargs=None):
        super().fit(data, convert_dataframe_kwargs, fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self

    @ParameterTypeAssert({
        'n': int,
        'predict_kwargs': (None, dict)
    })
    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        res = self.model.predict(n, **predict_kwargs).pd_dataframe()
        res = self.rename_prediction(res)
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
