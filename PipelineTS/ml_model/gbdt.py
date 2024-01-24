import pandas as pd
from darts.models import (
    CatBoostModel as CBT,
    LightGBMModel as LGB,
    XGBModel as XGB,
    RandomForest as RF
)
from spinesUtils.asserts import generate_function_kwargs, ParameterTypeAssert
from PipelineTS.base.base import GBDTModelMixin, IntervalEstimationMixin
from PipelineTS.base.darts_base import DartsForecastMixin
from PipelineTS.utils import check_time_col_is_timestamp


class CatBoostModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'lags_past_covariates': (None, int),
        'lags_future_covariates': (None, int),
        'add_encoders': (dict, None),
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
        """
        Initialize the CatBoostModel.

        Parameters
        ----------
        time_col : str
            The name of the time column in the input DataFrame.

        target_col : str
            The name of the target column in the input DataFrame.

        lags : int
            The number of lagged time steps to consider. Default is 1.

        lags_past_covariates : int or None, optional
            The number of lagged time steps for past covariates. Default is None.

        lags_future_covariates : int or None, optional
            The number of lagged time steps for future covariates. Default is None.

        add_encoders : dict or None, optional
            Additional encoders for categorical variables. Default is None.

        quantile : float or None, optional
            The quantile level for prediction intervals. Default is 0.9 (90%).

        random_state : int or None, optional
            The random seed for reproducibility. Default is None.

        multi_models : bool, optional
            Whether to use multiple models. Default is True.

        use_static_covariates : bool, optional
            Whether to use static covariates. Default is True.

        verbose : bool, optional
            Whether to print verbose output. Default is False.

        **darts_catboost_model_configs
            Additional configurations specific to the CatBoost model.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        # Generate model configurations
        self.all_configs['model_configs'] = generate_function_kwargs(
            CBT,
            lags=lags + 1,
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

        # Define and initialize the model
        self.model = self._define_model()

        # Update configurations
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
        """Define the CatBoost model."""
        return CBT(**self.all_configs['model_configs'])


class LightGBMModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'lags_past_covariates': (None, int),
        'lags_future_covariates': (None, int),
        'add_encoders': (dict, None),
        'quantile': (None, float),
        'random_state': (None, int),
        'multi_models': bool,
        'use_static_covariates': bool,
        'categorical_past_covariates': (str, list, None),
        'categorical_future_covariates': (str, list, None),
        'categorical_static_covariates': (str, list, None),
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
        """
        Initialize the LightGBMModel.

        Parameters
        ----------
        time_col : str
            The name of the time column in the input DataFrame.

        target_col : str
            The name of the target column in the input DataFrame.

        lags : int
            The number of lagged time steps to consider. Default is 1.

        lags_past_covariates : int or None, optional
            The number of lagged time steps for past covariates. Default is None.

        lags_future_covariates : int or None, optional
            The number of lagged time steps for future covariates. Default is None.

        add_encoders : dict or None, optional
            Additional encoders for categorical variables. Default is None.

        quantile : float or None, optional
            The quantile level for prediction intervals. Default is 0.9 (90%).

        random_state : int or None, optional
            The random seed for reproducibility. Default is None.

        multi_models : bool, optional
            Whether to use multiple models. Default is True.

        use_static_covariates : bool, optional
            Whether to use static covariates. Default is True.

        categorical_past_covariates : str, list, or None, optional
            Categorical past covariates. Default is None.

        categorical_future_covariates : str, list, or None, optional
            Categorical future covariates. Default is None.

        categorical_static_covariates : str, list, or None, optional
            Categorical static covariates. Default is None.

        verbose : int, optional
            Verbosity level. Default is -1.

        linear_tree : bool, optional
            Whether to use linear tree models. Default is True.

        **darts_lightgbm_model_configs
            Additional configurations specific to the LightGBM model.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        # Generate model configurations
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

        # Define and initialize the model
        self.model = self._define_model()

        # Update configurations
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
        """Define the LightGBM model."""
        return LGB(**self.all_configs['model_configs'])


class XGBoostModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'lags_past_covariates': (None, int),
        'lags_future_covariates': (None, int),
        'add_encoders': (dict, None),
        'quantile': (None, float),
        'random_state': (None, int),
        'multi_models': bool,
        'use_static_covariates': bool,
        'categorical_past_covariates': (str, list, None),
        'categorical_future_covariates': (str, list, None),
        'categorical_static_covariates': (str, list, None),
        'verbose': int
    }, 'XGBoostModel')
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
        """
        Initialize the XGBoostModel.

        Parameters
        ----------
        time_col : str
            The name of the time column in the input DataFrame.

        target_col : str
            The name of the target column in the input DataFrame.

        lags : int
            The number of lagged time steps to consider. Default is 1.

        lags_past_covariates : int or None, optional
            The number of lagged time steps for past covariates. Default is None.

        lags_future_covariates : int or None, optional
            The number of lagged time steps for future covariates. Default is None.

        add_encoders : dict or None, optional
            Additional encoders for categorical variables. Default is None.

        quantile : float or None, optional
            The quantile level for prediction intervals. Default is 0.9 (90%).

        random_state : int or None, optional
            The random seed for reproducibility. Default is None.

        multi_models : bool, optional
            Whether to use multiple models. Default is True.

        use_static_covariates : bool, optional
            Whether to use static covariates. Default is True.

        categorical_past_covariates : str, list, or None, optional
            Categorical past covariates. Default is None.

        categorical_future_covariates : str, list, or None, optional
            Categorical future covariates. Default is None.

        categorical_static_covariates : str, list, or None, optional
            Categorical static covariates. Default is None.

        verbose : int, optional
            Verbosity level. Default is 0.

        **darts_xgboost_model_configs
            Additional configurations specific to the XGBoost model.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        # Generate model configurations
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

        # Define and initialize the model
        self.model = self._define_model()

        # Update configurations
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
        """Define the XGBoost model."""
        return XGB(**self.all_configs['model_configs'])


class RandomForestModel(DartsForecastMixin, GBDTModelMixin, IntervalEstimationMixin):
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'lags_past_covariates': (None, int),
        'lags_future_covariates': (None, int),
        'add_encoders': (dict, None),
        'n_estimators': int,
        'quantile': (None, float),
        'random_state': (None, int),
        'multi_models': bool,
        'use_static_covariates': bool
    }, 'RandomForestModel')
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
        """
        Initialize the RandomForestModel.

        Parameters
        ----------
        time_col : str
            The name of the time column in the input DataFrame.

        target_col : str
            The name of the target column in the input DataFrame.

        lags : int
            The number of lagged time steps to consider.

        lags_past_covariates : int or None, optional
            The number of lagged time steps for past covariates. Default is None.

        lags_future_covariates : int or None, optional
            The number of lagged time steps for future covariates. Default is None.

        add_encoders : dict or None, optional
            Additional encoders for categorical variables. Default is None.

        n_estimators : int
            The number of trees in the forest.

        quantile : float or None, optional
            The quantile level for prediction intervals. Default is 0.9 (90%).

        random_state : int or None, optional
            The random seed for reproducibility. Default is None.

        multi_models : bool, optional
            Whether to use multiple models. Default is True.

        use_static_covariates : bool, optional
            Whether to use static covariates. Default is True.

        **darts_random_forest_model_configs
            Additional configurations specific to the RandomForest model.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        # Generate model configurations
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

        # Define and initialize the model
        self.model = self._define_model()

        # Update configurations
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
        """Define the RandomForest model."""
        return RF(**self.all_configs['model_configs'])

    @ParameterTypeAssert({
        'data': pd.DataFrame,
        'convert_dataframe_kwargs': (None, dict),
        'cv': int,
        'fit_kwargs': (None, dict)
    })
    def fit(self, data, cv=5, convert_dataframe_kwargs=None, fit_kwargs=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        cv : int, optional
            The number of cross-validation folds. Default is 5.

        convert_dataframe_kwargs : dict or None, optional
            Additional keyword arguments for converting the DataFrame. Default is None.

        fit_kwargs : dict or None, optional
            Additional keyword arguments for fitting the model. Default is None.

        Returns
        -------
        self
            Returns an instance of the fitted model.
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        super().fit(data=data, convert_dataframe_kwargs=convert_dataframe_kwargs,
                    fit_kwargs=fit_kwargs, valid_data=None)

        # Calculate quantile error if quantile is specified
        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self
