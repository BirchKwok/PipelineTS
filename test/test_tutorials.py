"""
Test cases extracted from all 12 PipelineTS tutorials.
Covers: datasets, single models, pipelines, preprocessing, metrics,
        multivariate, multi-series, covariates, incremental learning,
        SmartRouter, quantile intervals, save/load, and visualization.

Visualization tests use non-interactive backend (Agg) and only check
that figures are created without error (no display).

Optuna tests (tutorial 06) are kept lightweight (2 trials).
Chronos tests are skipped if chronos-forecasting is not installed.
"""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings('ignore')

# Use non-interactive matplotlib backend for CI
import matplotlib
matplotlib.use('Agg')


# ──────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def base_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = 50 + 10 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.randn(n) * 2
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture(scope='module')
def panel_data():
    np.random.seed(42)
    dfs = []
    for sid in ['A', 'B', 'C']:
        n = 150
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        v = np.random.uniform(30, 70) + 8 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 2
        dfs.append(pd.DataFrame({'date': dates, 'value': v, 'store': sid}))
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture(scope='module')
def multivariate_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    return pd.DataFrame({
        'date': dates,
        'value': np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1,
        'feature_a': np.cos(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1,
        'feature_b': np.sin(np.linspace(0, 2 * np.pi, n)) * 0.5 + np.random.randn(n) * 0.05,
    })


@pytest.fixture(scope='module')
def covariate_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    holiday = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    promotion = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    temperature = 15 + 10 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 2
    values = (50 + 10 * np.sin(np.linspace(0, 6 * np.pi, n))
              + 8 * holiday + 5 * promotion + 0.3 * temperature
              + np.random.randn(n) * 2)
    return pd.DataFrame({
        'date': dates, 'value': values,
        'holiday': holiday, 'promotion': promotion, 'temperature': temperature,
    })


LAGS = 12
PREDICT_N = 10


# ======================================================================
# Tutorial 01: QuickStart Guide
# ======================================================================

class TestTutorial01QuickStart:

    def test_load_datasets(self):
        from PipelineTS.dataset import (
            LoadElectric, LoadMessagesSent,
            LoadWebSales, LoadSupermarketIncoming
        )
        for loader in [LoadElectric, LoadMessagesSent,
                       LoadWebSales, LoadSupermarketIncoming]:
            df = loader()
            assert df.shape[0] > 0
            assert df.shape[1] >= 2

    def test_single_model_fit_predict(self, base_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS, quantile=0.9,
        )
        model.fit(base_data)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N
        assert 'value' in result.columns

    def test_pipeline_ml(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, include_models='ml', cv=3,
        )
        lb = pipeline.fit(base_data)
        assert len(lb) > 0
        result = pipeline.predict(20)
        assert result.shape[0] == 20
        assert 'value_lower' in result.columns
        assert 'value_upper' in result.columns

    def test_save_load_pipeline(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        from PipelineTS.io import save_model, load_model

        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'], quantile=None, cv=2,
        )
        pipeline.fit(base_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_pipe.pts')
            save_model(path, pipeline)
            loaded = load_model(path)
            result = loaded.predict(5)
            assert result.shape[0] == 5


# ======================================================================
# Tutorial 02: All Models Guide
# ======================================================================

class TestTutorial02AllModels:

    @pytest.mark.parametrize("model_cls_name", [
        'NLinearModel', 'DLinearModel', 'NBeatsModel', 'NHitsModel',
        'TFTModel', 'TransformerModel', 'TiDEModel', 'GAUModel',
        'StackingRNNModel', 'Time2VecModel', 'PatchRNNModel',
        'TCNModel', 'ITransformerModel', 'SRSNetModel', 'DeepARModel',
    ])
    def test_nn_model(self, base_data, model_cls_name):
        import PipelineTS.nn_model as nn_mod
        ModelClass = getattr(nn_mod, model_cls_name)
        model = ModelClass(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=30, patience=5, verbose=False,
        )
        model.fit(base_data)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N

    def test_wide_gbrt(self, base_data):
        from PipelineTS.ml_model import WideGBRTModel
        model = WideGBRTModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, n_estimators=200, verbose=False, differential_n=1,
        )
        model.fit(base_data)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N

    def test_multi_output_models(self, base_data):
        from PipelineTS.ml_model import (
            MultiOutputRegressorModel, MultiStepRegressorModel,
            RegressorChainModel,
        )
        for ModelClass in [MultiOutputRegressorModel, MultiStepRegressorModel,
                           RegressorChainModel]:
            model = ModelClass(
                time_col='date', target_col='value', lags=LAGS,
                quantile=0.9, verbose=False,
            )
            model.fit(base_data)
            result = model.predict(PREDICT_N)
            assert result.shape[0] == PREDICT_N

    def test_prophet(self, base_data):
        from PipelineTS.statistic_model import ProphetModel
        model = ProphetModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, auto_seasonality=True,
        )
        model.fit(base_data, cv=2)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N

    def test_auto_arima(self, base_data):
        from PipelineTS.statistic_model import AutoARIMAModel
        model = AutoARIMAModel(
            time_col='date', target_col='value', lags=LAGS,
            start_p=0, max_p=3, start_q=0, max_q=3,
            seasonal=False, quantile=0.9,
        )
        model.fit(base_data, cv=2)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N


# ======================================================================
# Tutorial 03: Multivariate Prediction
# ======================================================================

class TestTutorial03Multivariate:

    def test_itransformer_univariate(self, multivariate_data):
        from PipelineTS.nn_model import ITransformerModel
        model = ITransformerModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=32, n_heads=2, d_ff=64, e_layers=1,
            quantile=None, epochs=30, patience=5, verbose=False,
        )
        model.fit(multivariate_data)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N

    def test_itransformer_multivariate(self, multivariate_data):
        from PipelineTS.nn_model import ITransformerModel
        model = ITransformerModel(
            time_col='date', target_col='value',
            feature_cols=['value', 'feature_a', 'feature_b'],
            lags=LAGS, d_model=32, n_heads=2, d_ff=64, e_layers=1,
            quantile=None, epochs=30, patience=5, verbose=False,
        )
        model.fit(multivariate_data)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N

    def test_srsnet_multivariate(self, multivariate_data):
        from PipelineTS.nn_model import SRSNetModel
        model = SRSNetModel(
            time_col='date', target_col='value',
            feature_cols=['value', 'feature_a', 'feature_b'],
            lags=LAGS, d_model=32, n_heads=2,
            quantile=None, epochs=30, patience=5, verbose=False,
        )
        model.fit(multivariate_data)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N

    def test_pipeline_multivariate(self, multivariate_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value',
            feature_cols=['value', 'feature_a', 'feature_b'],
            lags=LAGS,
            include_models=['catboost', 'random_forest'],
            quantile=None, cv=2,
        )
        lb = pipeline.fit(multivariate_data)
        assert len(lb) > 0


# ======================================================================
# Tutorial 04: Advanced Pipeline
# ======================================================================

class TestTutorial04AdvancedPipeline:

    def test_pipeline_configs(self, base_data):
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_small', {
                'init_configs': {'iterations': 32}, 'fit_configs': {},
            }),
            ('catboost', 'boost_large', {
                'init_configs': {'iterations': 128}, 'fit_configs': {},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2,
        )
        lb = pipeline.fit(base_data)
        assert len(lb) == 2

    def test_per_model_lags(self, base_data):
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_lag6', {
                'init_configs': {'iterations': 32},
                'pipeline_configs': {'lags': 6},
            }),
            ('catboost', 'boost_lag20', {
                'init_configs': {'iterations': 32},
                'pipeline_configs': {'lags': 20},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2,
        )
        pipeline.fit(base_data)
        m6 = pipeline.get_model('boost_lag6')
        m20 = pipeline.get_model('boost_lag20')
        assert m6.all_configs['lags'] == 6
        assert m20.all_configs['lags'] == 20

    def test_per_model_scaler(self, base_data):
        from sklearn.preprocessing import StandardScaler
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_standard', {
                'init_configs': {'iterations': 32},
                'pipeline_configs': {'scaler': StandardScaler()},
            }),
            ('catboost', 'boost_noscale', {
                'init_configs': {'iterations': 32},
                'pipeline_configs': {'scaler': None},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, scaler=True, quantile=None, cv=2,
        )
        pipeline.fit(base_data)
        assert pipeline._model_scalers.get('boost_noscale') is None

    def test_model_filtering_ml(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models='ml', quantile=None, cv=2,
        )
        lb = pipeline.fit(base_data)
        assert len(lb) > 0

    def test_custom_metric(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        from PipelineTS.spinesTS.metrics import rmse
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost', 'random_forest'],
            metric=rmse, metric_less_is_better=True,
            quantile=None, cv=2,
        )
        lb = pipeline.fit(base_data)
        assert len(lb) == 2

    def test_double_underscore_syntax(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost', 'random_forest'],
            quantile=None, cv=2,
            catboost__iterations=64,
            random_forest__n_estimators=128,
        )
        lb = pipeline.fit(base_data)
        assert len(lb) == 2

    def test_get_model_and_configs(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'], quantile=None, cv=2,
        )
        pipeline.fit(base_data)
        best = pipeline.get_model()
        assert best is not None
        configs = pipeline.get_model_all_configs()
        assert 'lags' in configs


# ======================================================================
# Tutorial 05: Preprocessing and Data
# ======================================================================

class TestTutorial05Preprocessing:

    def test_builtin_datasets(self):
        from PipelineTS.dataset import (
            LoadElectric, LoadMessagesSentHour,
            LoadMessagesSent, LoadWebSales, LoadSupermarketIncoming,
        )
        for loader in [LoadElectric, LoadMessagesSentHour,
                       LoadMessagesSent, LoadWebSales,
                       LoadSupermarketIncoming]:
            df = loader()
            assert df.shape[0] > 0

    def test_data_generator(self):
        from PipelineTS.dataset import DataGenerator
        gen = DataGenerator()
        synthetic = gen.trigonometry_ds(size=200)
        assert len(synthetic) == 200

    def test_scalers(self):
        from PipelineTS.preprocessing import Scaler
        X = np.random.randn(100, 1)
        for name in ['min_max', 'standard', 'quantile', 'gauss_rank']:
            scaler = Scaler(name)
            transformed = scaler.fit_transform(X)
            recovered = scaler.inverse_transform(transformed)
            assert transformed.shape == X.shape
            assert recovered.shape == X.shape

    def test_split_series(self):
        from PipelineTS.spinesTS.preprocessing import (
            split_series, train_test_split_ts,
        )
        series = np.sin(np.linspace(0, 4 * np.pi, 100))
        X, y = split_series(series, series, window_size=10, pred_steps=5)
        assert X.shape[1] == 10
        assert y.shape[1] == 5
        X_tr, X_te, y_tr, y_te = train_test_split_ts(X, y, train_size=0.8)
        assert X_tr.shape[0] + X_te.shape[0] == X.shape[0]

    def test_split_series_multivariate(self):
        from PipelineTS.spinesTS.preprocessing import split_series_multivariate
        multi = np.random.randn(100, 3).astype(np.float32)
        X, y = split_series_multivariate(multi, multi, window_size=10, pred_steps=5)
        assert X.ndim == 3
        assert X.shape[2] == 3

    def test_metrics(self):
        from PipelineTS.spinesTS.metrics import mae, mse, rmse, wmape
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yp = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        assert mae(yt, yp) > 0
        assert mse(yt, yp) > 0
        assert rmse(yt, yp) > 0
        assert wmape(yt, yp) > 0

    def test_quantile_acc(self):
        from PipelineTS.metrics import quantile_acc
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        hi = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        acc = quantile_acc(yt, lo, hi)
        assert acc == 1.0

    def test_builtin_series_data(self):
        from PipelineTS.dataset import BuiltInSeriesData
        sd = BuiltInSeriesData()
        etth1 = sd['ETTh1']
        assert etth1.shape[0] > 0


# ======================================================================
# Tutorial 06: Hyperparameter Tuning (lightweight, 2 trials)
# ======================================================================

class TestTutorial06HyperparameterTuning:

    def test_optuna_ml(self):
        optuna = pytest.importorskip('optuna')
        from PipelineTS.dataset import LoadMessagesSent
        from PipelineTS.pipeline import ModelPipeline
        from PipelineTS.ml_model import WideGBRTModel
        from sklearn.metrics import mean_absolute_error

        init_data = LoadMessagesSent()
        init_data = init_data[['date', 'ta']]
        init_data['date'] = pd.to_datetime(init_data['date'])
        n = 30
        valid_data = init_data.iloc[-n:]
        data = init_data.iloc[:-n]

        def objective(trial):
            lags = trial.suggest_int('lags', 8, 30, step=2)
            pipeline = ModelPipeline(
                time_col='date', target_col='ta', lags=lags,
                random_state=42, include_models=WideGBRTModel,
                metric=mean_absolute_error, metric_less_is_better=True,
                scaler=None,
            )
            pipeline.fit(data, valid_data=valid_data)
            pred = pipeline.predict(n)
            return mean_absolute_error(valid_data['ta'].values, pred['ta'].values)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=2)
        assert study.best_value > 0

    def test_optuna_nn(self):
        optuna = pytest.importorskip('optuna')
        from PipelineTS.dataset import LoadMessagesSent
        from PipelineTS.nn_model import TCNModel
        from sklearn.metrics import mean_absolute_error

        init_data = LoadMessagesSent()
        init_data = init_data[['date', 'ta']]
        init_data['date'] = pd.to_datetime(init_data['date'])
        n = 30
        valid_data = init_data.iloc[-n:]
        data = init_data.iloc[:-n]

        def objective(trial):
            lags = trial.suggest_int('lags', 8, 24, step=4)
            model = TCNModel(
                time_col='date', target_col='ta', lags=lags,
                epochs=30, patience=10, verbose=False, random_state=42,
            )
            model.fit(data)
            pred = model.predict(n)
            return mean_absolute_error(valid_data['ta'].values, pred['ta'].values)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=2)
        assert study.best_value > 0


# ======================================================================
# Tutorial 07: Benchmarks
# ======================================================================

class TestTutorial07Benchmarks:

    def test_multi_dataset_benchmark(self):
        from PipelineTS.dataset import LoadElectric, LoadWebSales
        from PipelineTS.pipeline import ModelPipeline
        from sklearn.metrics import mean_absolute_error

        datasets = {
            'Electric': (LoadElectric, 'date', 'value'),
            'WebSales': (LoadWebSales, 'date', 'type_a'),
        }
        for name, (loader, tc, tgt) in datasets.items():
            df = loader()[[tc, tgt]]
            df[tc] = pd.to_datetime(df[tc])
            pipe = ModelPipeline(
                time_col=tc, target_col=tgt, lags=LAGS,
                random_state=42,
                include_models=['catboost'],
                metric=mean_absolute_error, metric_less_is_better=True,
                quantile=None, cv=2,
            )
            lb = pipe.fit(df)
            assert len(lb) > 0

    def test_benchmark_with_quantile(self):
        from PipelineTS.dataset import LoadElectric
        from PipelineTS.pipeline import ModelPipeline
        from sklearn.metrics import mean_absolute_error

        df = LoadElectricProduction()
        df['date'] = pd.to_datetime(df['date'])
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            random_state=42, include_models=['catboost'],
            metric=mean_absolute_error, metric_less_is_better=True,
            quantile=0.9, cv=2,
        )
        lb = pipe.fit(df)
        assert 'quantile_acc' in lb.columns


# ======================================================================
# Tutorial 08: Visualization
# ======================================================================

class TestTutorial08Visualization:

    def test_configure_chinese_font(self):
        from PipelineTS.plot import configure_chinese_font
        font_name = configure_chinese_font()
        # May return None on systems without Chinese fonts, but should not error

    def test_plot_series(self, base_data):
        from PipelineTS.plot import plot_series
        fig = plot_series(base_data, time_col='date', target_col='value',
                          title='test', show=False)
        assert fig is not None

    def test_plot_series_panel(self, panel_data):
        from PipelineTS.plot import plot_series
        fig = plot_series(panel_data, time_col='date', target_col='value',
                          id_col='store', title='panel', show=False)
        assert fig is not None

    def test_plot_forecast(self, base_data):
        from PipelineTS.ml_model import CatBoostModel
        from PipelineTS.plot import plot_forecast
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS, quantile=0.9,
        )
        model.fit(base_data)
        pred = model.predict(20)
        fig = plot_forecast(base_data, pred, time_col='date', target_col='value',
                            history_tail=60, show=False)
        assert fig is not None

    def test_plot_leaderboard(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        from PipelineTS.plot import plot_leaderboard, plot_leaderboard_detail
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost', 'random_forest'],
            quantile=0.9, cv=2,
        )
        lb = pipe.fit(base_data)
        fig1 = plot_leaderboard(lb, title='test', show=False)
        fig2 = plot_leaderboard_detail(lb, title='test', show=False)
        assert fig1 is not None
        assert fig2 is not None

    def test_plot_residuals(self, base_data):
        from PipelineTS.plot import plot_residuals
        yt = base_data['value'].values[-50:]
        yp = yt + np.random.randn(50) * 2
        fig = plot_residuals(yt, yp, time_index=base_data['date'].values[-50:],
                             show=False)
        assert fig is not None

    def test_plot_acf_pacf(self, base_data):
        from PipelineTS.plot import plot_acf_pacf
        fig = plot_acf_pacf(base_data['value'].values, max_lags=30, show=False)
        assert fig is not None

    def test_plot_decomposition(self, base_data):
        from PipelineTS.plot import plot_decomposition
        fig = plot_decomposition(base_data, time_col='date', target_col='value',
                                 model='additive', show=False)
        assert fig is not None

    def test_plot_train_test_split(self, base_data):
        from PipelineTS.plot import plot_train_test_split
        train = base_data.iloc[:-30]
        test = base_data.iloc[-30:]
        fig = plot_train_test_split(train, test, time_col='date', target_col='value',
                                    show=False)
        assert fig is not None

    def test_tsplotter(self, base_data):
        from PipelineTS.plot import TSPlotter
        plotter = TSPlotter(time_col='date', target_col='value', lang='en')
        fig = plotter.plot_series(base_data, title='test', show=False)
        assert fig is not None

    def test_pipeline_plot(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=0.9, cv=2,
        )
        pipe.fit(base_data)
        fig = pipe.plot(n=15, history_tail=60, show=False)
        assert fig is not None

    def test_colors(self):
        from PipelineTS.plot import COLORS, MODEL_COLORS
        assert len(COLORS) > 0
        assert len(MODEL_COLORS) > 0


# ======================================================================
# Tutorial 09: Multi-Quantile Intervals
# ======================================================================

class TestTutorial09MultiQuantile:

    def test_single_quantile(self, base_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS, quantile=0.9,
        )
        model.fit(base_data)
        result = model.predict(15)
        assert 'value_lower' in result.columns
        assert 'value_upper' in result.columns

    def test_multi_quantile_pipeline(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS, quantile=0.9,
            include_models=['catboost', 'random_forest'],
            cv=2,
        )
        pipe.fit(base_data)
        result = pipe.predict_quantiles(n=15, levels=[0.5, 0.8, 0.95])
        assert 'value_q0.5_lower' in result.columns
        assert 'value_q0.95_upper' in result.columns

    def test_monotonicity(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS, quantile=0.9,
            include_models=['catboost'], cv=2,
        )
        pipe.fit(base_data)
        result = pipe.predict_quantiles(n=10, levels=[0.5, 0.8, 0.95])
        w50 = result['value_q0.5_upper'] - result['value_q0.5_lower']
        w80 = result['value_q0.8_upper'] - result['value_q0.8_lower']
        w95 = result['value_q0.95_upper'] - result['value_q0.95_lower']
        assert np.all(w80.values >= w50.values - 1e-6)
        assert np.all(w95.values >= w80.values - 1e-6)

    def test_nn_cqr(self, base_data):
        from PipelineTS.nn_model import NLinearModel
        model = NLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=30, patience=5, verbose=False,
        )
        model.fit(base_data)
        result = model.predict(15)
        assert 'value_lower' in result.columns

    def test_smartrouter_multi_quantile(self, base_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value',
            quantile=0.9, max_models=2,
        )
        router.fit(base_data)
        result = router.predict_quantiles(n=10, levels=[0.5, 0.9])
        assert 'value_q0.5_lower' in result.columns


# ======================================================================
# Tutorial 10: Multi-Series & Covariates
# ======================================================================

class TestTutorial10MultiSeriesCovariates:

    def test_single_model_panel(self, panel_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS, quantile=0.9,
        )
        model.all_configs['id_col'] = 'store'
        model.fit(panel_data)
        result = model.predict(PREDICT_N)
        assert 'store' in result.columns
        assert result.groupby('store').size().min() == PREDICT_N

    def test_pipeline_panel(self, panel_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            id_col='store',
            include_models=['catboost'],
            quantile=0.9, cv=2,
        )
        lb = pipe.fit(panel_data)
        assert len(lb) > 0
        pred = pipe.predict(n=PREDICT_N)
        assert 'store' in pred.columns
        assert pred.groupby('store').size().min() == PREDICT_N

    def test_smartrouter_panel(self, panel_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value',
            id_col='store', max_models=2,
        )
        router.fit(panel_data)
        pred = router.predict(5)
        assert 'store' in pred.columns

    def test_pipeline_known_covariates(self, covariate_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            known_covariates=['holiday', 'promotion'],
            past_covariates=['temperature'],
            include_models=['catboost', 'prophet'],
            quantile=0.9, cv=2,
        )
        pipe.fit(covariate_data)
        future = pd.DataFrame({
            'holiday': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            'promotion': [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        })
        pred = pipe.predict(n=PREDICT_N, future_covariates=future)
        assert pred.shape[0] == PREDICT_N

    def test_pipeline_no_future_covariates(self, covariate_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            known_covariates=['holiday'],
            include_models=['catboost'],
            quantile=None, cv=2,
        )
        pipe.fit(covariate_data)
        pred = pipe.predict(n=5)
        assert pred.shape[0] == 5

    def test_smartrouter_covariates(self, covariate_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value',
            known_covariates=['holiday', 'promotion'],
            past_covariates=['temperature'],
            max_models=2,
        )
        router.fit(covariate_data)
        future = pd.DataFrame({
            'holiday': [0, 0, 0, 1, 0],
            'promotion': [1, 0, 0, 0, 0],
        })
        pred = router.predict(5, future_covariates=future)
        assert pred.shape[0] == 5

    def test_combined_panel_covariates(self):
        np.random.seed(42)
        dfs = []
        for sid in ['North', 'South']:
            n = 150
            dates = pd.date_range('2020-01-01', periods=n, freq='D')
            holiday = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
            vals = np.random.uniform(40, 70) + 8 * np.sin(np.linspace(0, 4 * np.pi, n)) \
                   + 6 * holiday + np.random.randn(n) * 2
            dfs.append(pd.DataFrame({
                'date': dates, 'value': vals, 'region': sid, 'holiday': holiday,
            }))
        combined = pd.concat(dfs, ignore_index=True)

        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            id_col='region', known_covariates=['holiday'],
            include_models=['catboost'],
            quantile=0.9, cv=2,
        )
        pipe.fit(combined)
        future = pd.DataFrame({'holiday': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]})
        pred = pipe.predict(n=PREDICT_N, future_covariates=future)
        assert 'region' in pred.columns


# ======================================================================
# Tutorial 11: Incremental Learning
# ======================================================================

class TestTutorial11IncrementalLearning:

    def test_pipeline_update(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        initial = base_data.iloc[:150].copy()
        batch = base_data.iloc[150:].copy()

        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'], quantile=None, cv=2,
        )
        pipe.fit(initial)
        pred_before = pipe.predict(3)

        pipe.update(batch)
        pred_after = pipe.predict(3)

        assert pred_before.shape[0] == 3
        assert pred_after.shape[0] == 3
        # Predictions should differ after update
        assert not np.allclose(pred_before['value'].values, pred_after['value'].values)

    def test_smartrouter_update(self, base_data):
        from PipelineTS.pipeline import SmartRouter
        initial = base_data.iloc[:150].copy()
        batch = base_data.iloc[150:].copy()

        router = SmartRouter(
            time_col='date', target_col='value', max_models=2,
        )
        router.fit(initial)
        pred_before = router.predict(3)

        router.update(batch)
        pred_after = router.predict(3)

        assert pred_before.shape[0] == 3
        assert pred_after.shape[0] == 3

    def test_nn_warmstart_update(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        initial = base_data.iloc[:150].copy()
        batch = base_data.iloc[150:].copy()

        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['n_linear'],
            quantile=None, cv=2,
            n_linear__epochs=30, n_linear__patience=5, n_linear__verbose=False,
        )
        pipe.fit(initial)
        pipe.update(batch)
        pred = pipe.predict(5)
        assert pred.shape[0] == 5

    def test_streaming_simulation(self, base_data):
        from PipelineTS.pipeline import ModelPipeline

        n = 300
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        vals = 50 + 10 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.randn(n) * 2
        full = pd.DataFrame({'date': dates, 'value': vals})

        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'], quantile=None, cv=2,
        )
        pipe.fit(full.iloc[:200])

        for start in [200, 233, 266]:
            end = min(start + 33, n)
            pipe.update(full.iloc[start:end])

        pred = pipe.predict(3)
        assert pred.shape[0] == 3

    def test_update_unfitted_raises(self):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
        )
        with pytest.raises(ValueError):
            pipe.update(pd.DataFrame({'date': [1], 'value': [1]}))


# ======================================================================
# Tutorial 12: SmartRouter & Pipeline Deep Dive
# ======================================================================

class TestTutorial12SmartRouterPipeline:

    def test_pipeline_basic_ml(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=12,
            include_models='ml', quantile=0.9, cv=3,
        )
        lb = pipe.fit(base_data)
        assert len(lb) > 0
        result = pipe.predict(n=15)
        assert result.shape[0] == 15
        assert 'value_lower' in result.columns

    def test_list_all_models(self):
        from PipelineTS.pipeline import ModelPipeline
        models = ModelPipeline.list_all_available_models()
        assert len(models) >= 20

    def test_pipeline_configs_variants(self, base_data):
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_fast', {
                'init_configs': {'iterations': 16}, 'fit_configs': {},
            }),
            ('catboost', 'boost_deep', {
                'init_configs': {'iterations': 128, 'depth': 7}, 'fit_configs': {},
            }),
        ])
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=12,
            include_models=['catboost'], configs=configs,
            quantile=None, cv=2,
        )
        lb = pipe.fit(base_data)
        assert len(lb) == 2

    def test_multi_quantile_output(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=12, quantile=0.9,
            include_models=['catboost'], cv=2,
        )
        pipe.fit(base_data)
        result = pipe.predict_quantiles(n=10, levels=[0.5, 0.8, 0.95])
        assert 'value_q0.5_lower' in result.columns
        assert 'value_q0.95_upper' in result.columns

    def test_pipeline_panel(self, panel_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=12,
            id_col='store',
            include_models=['catboost'],
            quantile=0.9, cv=2,
        )
        pipe.fit(panel_data)
        pred = pipe.predict(n=5)
        assert 'store' in pred.columns
        assert pred.shape[0] == 15  # 5 per series * 3 series

    def test_pipeline_covariates(self, base_data):
        cov_data = base_data.copy()
        np.random.seed(42)
        cov_data['holiday'] = np.random.choice([0, 1], size=len(base_data), p=[0.9, 0.1])
        cov_data['temperature'] = 15 + 10 * np.sin(np.linspace(0, 2 * np.pi, len(base_data))) \
                                  + np.random.randn(len(base_data))

        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=12,
            known_covariates=['holiday'], past_covariates=['temperature'],
            include_models=['catboost', 'prophet'],
            quantile=0.9, cv=2,
        )
        pipe.fit(cov_data)
        future = pd.DataFrame({'holiday': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]})
        pred = pipe.predict(n=10, future_covariates=future)
        assert pred.shape[0] == 10

    def test_pipeline_incremental(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=12,
            include_models=['catboost'], quantile=None, cv=2,
        )
        pipe.fit(base_data.iloc[:150])
        pipe.update(base_data.iloc[150:])
        pred = pipe.predict(3)
        assert pred.shape[0] == 3

    def test_pipeline_time_budget(self, base_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=12,
            include_models='ml', time_limit=30,
            quantile=None, cv=2,
        )
        pipe.fit(base_data)
        assert len(pipe.leader_board_) > 0

    def test_smartrouter_basic(self, base_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(time_col='date', target_col='value')
        router.fit(base_data)
        pred = router.predict(n=15)
        assert pred.shape[0] == 15

    def test_smartrouter_preset_fast(self, base_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value', preset='fast',
        )
        router.fit(base_data)
        pred = router.predict(10)
        assert pred.shape[0] == 10

    def test_smartrouter_panel(self, panel_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value',
            id_col='store', preset='fast',
        )
        router.fit(panel_data)
        pred = router.predict(5)
        assert 'store' in pred.columns
        assert pred.shape[0] == 15

    def test_smartrouter_covariates(self, base_data):
        cov_data = base_data.copy()
        np.random.seed(42)
        cov_data['holiday'] = np.random.choice([0, 1], size=len(base_data), p=[0.9, 0.1])
        cov_data['temperature'] = 15 + np.random.randn(len(base_data))

        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value',
            known_covariates=['holiday'], past_covariates=['temperature'],
            preset='fast',
        )
        router.fit(cov_data)
        future = pd.DataFrame({'holiday': [0, 0, 0, 1, 0]})
        pred = router.predict(5, future_covariates=future)
        assert pred.shape[0] == 5

    def test_smartrouter_thorough(self, base_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value',
            search_strategy='thorough', max_models=4,
        )
        router.fit(base_data)
        if router._screening_results is not None and not router._screening_results.empty:
            assert 'model' in router._screening_results.columns
        assert len(router.leader_board_) > 0

    def test_smartrouter_update(self, base_data):
        from PipelineTS.pipeline import SmartRouter
        router = SmartRouter(
            time_col='date', target_col='value', max_models=2,
        )
        router.fit(base_data.iloc[:150])
        router.update(base_data.iloc[150:])
        pred = router.predict(5)
        assert pred.shape[0] == 5

    def test_smartrouter_production(self):
        from PipelineTS.pipeline import SmartRouter
        from PipelineTS.dataset import LoadElectric

        electric = LoadElectricProduction()
        electric['date'] = pd.to_datetime(electric['date'])

        router = SmartRouter(
            time_col='date', target_col='value',
            preset='fast', quantile=0.9,
            max_models=2,
        )
        router.fit(electric)
        forecast = router.predict(n=12)
        assert forecast.shape[0] == 12
        assert 'value_lower' in forecast.columns


# ======================================================================
# Tutorial: Chronos Foundation Models (optional dependency)
# ======================================================================

class TestTutorialChronos:

    @pytest.fixture(autouse=True)
    def _skip_if_no_chronos(self):
        pytest.importorskip('chronos')

    def test_chronos_small(self, base_data):
        from PipelineTS.nn_model import Chronos2SmallModel
        model = Chronos2SmallModel(
            time_col='date', target_col='value', quantile=0.9,
        )
        model.fit(base_data, cv=2)
        result = model.predict(PREDICT_N)
        assert result.shape[0] == PREDICT_N
