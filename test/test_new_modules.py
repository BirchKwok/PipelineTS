"""
Comprehensive test suite for all new PipelineTS modules:
- Preprocessing: missing handler, outlier handler, data quality report,
  stationarity tests, frequency detection, time series split
- Metrics: mape, smape, mase, r2_score, medae, picp, pinaw, winkler_score
- Feature engineering: Fourier, holiday, lag, unified pipeline
- Evaluation: backtesting, residual analysis, model comparison
- Training: auto tune, ensemble
- Prediction: rolling predict, explainability
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_ts_data(n=200, freq='D', seed=42):
    """Create a simple time series DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq=freq)
    trend = np.linspace(100, 200, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 30)
    noise = np.random.randn(n) * 3
    values = trend + seasonal + noise
    return pd.DataFrame({'date': dates, 'value': values})


def make_ts_data_with_gaps(n=200, n_gaps=5, n_nans=3, seed=42):
    """Create time series with implicit gaps and explicit NaNs."""
    df = make_ts_data(n, seed=seed)
    rng = np.random.RandomState(seed)
    # Remove some rows (implicit gaps)
    drop_idx = rng.choice(range(10, n - 10), size=n_gaps, replace=False)
    df = df.drop(drop_idx).reset_index(drop=True)
    # Add explicit NaNs
    nan_idx = rng.choice(range(5, len(df) - 5), size=n_nans, replace=False)
    df.loc[nan_idx, 'value'] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  P0: PREPROCESSING — Missing Handler
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingHandler:
    def test_detect_no_missing(self):
        from PipelineTS.preprocessing import TimeSeriesMissingHandler
        df = make_ts_data(50)
        handler = TimeSeriesMissingHandler(time_col='date')
        report = handler.fit(df)
        assert report['n_implicit_gaps'] == 0
        assert len(report['n_explicit_nan']) == 0
        assert report['completeness_ratio'] == 1.0

    def test_detect_with_gaps(self):
        from PipelineTS.preprocessing import TimeSeriesMissingHandler
        df = make_ts_data_with_gaps(100, n_gaps=5, n_nans=3)
        handler = TimeSeriesMissingHandler(time_col='date')
        report = handler.fit(df)
        assert report['n_implicit_gaps'] >= 1
        assert report['n_explicit_nan'].get('value', 0) >= 1

    def test_fill_linear(self):
        from PipelineTS.preprocessing import TimeSeriesMissingHandler
        df = make_ts_data_with_gaps(100, n_gaps=3, n_nans=3)
        handler = TimeSeriesMissingHandler(time_col='date')
        filled = handler.transform(df, method='linear')
        assert filled['value'].isna().sum() == 0
        assert len(filled) >= len(df)

    def test_fill_ffill(self):
        from PipelineTS.preprocessing import TimeSeriesMissingHandler
        df = make_ts_data_with_gaps(100, n_gaps=0, n_nans=5)
        handler = TimeSeriesMissingHandler(time_col='date')
        filled = handler.transform(df, method='ffill', fill_implicit_gaps=False)
        assert filled['value'].isna().sum() == 0

    def test_fill_bfill(self):
        from PipelineTS.preprocessing import TimeSeriesMissingHandler
        df = make_ts_data_with_gaps(100, n_gaps=0, n_nans=5)
        handler = TimeSeriesMissingHandler(time_col='date')
        filled = handler.transform(df, method='bfill', fill_implicit_gaps=False)
        assert filled['value'].isna().sum() == 0

    def test_fill_zero(self):
        from PipelineTS.preprocessing import TimeSeriesMissingHandler
        df = make_ts_data_with_gaps(100, n_gaps=0, n_nans=3)
        handler = TimeSeriesMissingHandler(time_col='date')
        filled = handler.transform(df, method='zero', fill_implicit_gaps=False)
        assert filled['value'].isna().sum() == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  P0: PREPROCESSING — Outlier Handler
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutlierDetector:
    def _make_data_with_outliers(self):
        df = make_ts_data(100)
        df.loc[10, 'value'] = 1000  # extreme outlier
        df.loc[50, 'value'] = -500  # extreme outlier
        return df

    def test_detect_iqr(self):
        from PipelineTS.preprocessing import TimeSeriesOutlierDetector
        df = self._make_data_with_outliers()
        detector = TimeSeriesOutlierDetector(time_col='date', method='iqr')
        mask = detector.fit(df, target_col='value')
        assert mask['value'].sum() >= 2

    def test_detect_zscore(self):
        from PipelineTS.preprocessing import TimeSeriesOutlierDetector
        df = self._make_data_with_outliers()
        detector = TimeSeriesOutlierDetector(time_col='date', method='zscore')
        mask = detector.fit(df, target_col='value')
        assert mask['value'].sum() >= 2

    def test_detect_rolling_zscore(self):
        from PipelineTS.preprocessing import TimeSeriesOutlierDetector
        df = self._make_data_with_outliers()
        detector = TimeSeriesOutlierDetector(time_col='date', method='rolling_zscore', window=20)
        mask = detector.fit(df, target_col='value')
        assert mask['value'].sum() >= 1

    def test_handle_clip(self):
        from PipelineTS.preprocessing import TimeSeriesOutlierDetector
        df = self._make_data_with_outliers()
        detector = TimeSeriesOutlierDetector(time_col='date', method='iqr')
        cleaned = detector.transform(df, target_col='value', strategy='clip')
        assert cleaned['value'].max() < 500
        assert cleaned['value'].min() > -300

    def test_handle_nan(self):
        from PipelineTS.preprocessing import TimeSeriesOutlierDetector
        df = self._make_data_with_outliers()
        detector = TimeSeriesOutlierDetector(time_col='date', method='iqr')
        cleaned = detector.transform(df, target_col='value', strategy='nan')
        assert cleaned['value'].isna().sum() >= 2

    def test_handle_linear(self):
        from PipelineTS.preprocessing import TimeSeriesOutlierDetector
        df = self._make_data_with_outliers()
        detector = TimeSeriesOutlierDetector(time_col='date', method='iqr')
        cleaned = detector.transform(df, target_col='value', strategy='linear')
        assert cleaned['value'].isna().sum() == 0
        assert cleaned['value'].max() < 500


# ═══════════════════════════════════════════════════════════════════════════════
#  P0: PREPROCESSING — Data Quality Report
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataQualityReport:
    def test_generate_clean_data(self):
        from PipelineTS.preprocessing import TimeSeriesDataQualityReport
        df = make_ts_data(100)
        report_gen = TimeSeriesDataQualityReport(time_col='date', target_col='value')
        report = report_gen.fit(df)
        assert 'overview' in report
        assert 'time_analysis' in report
        assert 'value_analysis' in report
        assert 'missing_analysis' in report
        assert 'issues' in report
        assert report['overview']['n_rows'] == 100

    def test_generate_dirty_data(self):
        from PipelineTS.preprocessing import TimeSeriesDataQualityReport
        df = make_ts_data_with_gaps(100, n_gaps=5, n_nans=3)
        report_gen = TimeSeriesDataQualityReport(time_col='date', target_col='value')
        report = report_gen.fit(df)
        assert len(report['issues']) > 0

    def test_print_report(self, capsys):
        from PipelineTS.preprocessing import TimeSeriesDataQualityReport
        df = make_ts_data(50)
        report_gen = TimeSeriesDataQualityReport(time_col='date', target_col='value')
        report_gen.report(df)
        captured = capsys.readouterr()
        assert 'OVERVIEW' in captured.out
        assert 'TIME ANALYSIS' in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
#  P0: METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def _make_data(self):
        np.random.seed(42)
        y_true = np.array([100, 200, 300, 400, 500], dtype=np.float64)
        y_pred = np.array([110, 190, 310, 390, 510], dtype=np.float64)
        return y_true, y_pred

    def test_mape(self):
        from PipelineTS.metrics import mape
        yt, yp = self._make_data()
        result = mape(yt, yp)
        assert 0 < result < 1
        assert isinstance(result, float)

    def test_smape(self):
        from PipelineTS.metrics import smape
        yt, yp = self._make_data()
        result = smape(yt, yp)
        assert 0 <= result <= 2
        assert isinstance(result, float)

    def test_mase(self):
        from PipelineTS.metrics import mase
        yt, yp = self._make_data()
        y_train = np.array([50, 80, 120, 160, 200, 250, 300, 350, 400], dtype=np.float64)
        result = mase(yt, yp, y_train)
        assert isinstance(result, float)
        assert result > 0

    def test_r2_score(self):
        from PipelineTS.metrics import r2_score
        yt, yp = self._make_data()
        result = r2_score(yt, yp)
        assert isinstance(result, float)
        assert result > 0.9  # good prediction

    def test_r2_score_perfect(self):
        from PipelineTS.metrics import r2_score
        yt = np.array([1.0, 2.0, 3.0])
        result = r2_score(yt, yt)
        assert abs(result - 1.0) < 1e-10

    def test_medae(self):
        from PipelineTS.metrics import medae
        yt, yp = self._make_data()
        result = medae(yt, yp)
        assert isinstance(result, float)
        assert result == 10.0

    def test_picp(self):
        from PipelineTS.metrics import picp
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = picp(yt, lower, upper)
        assert result == 1.0

    def test_pinaw(self):
        from PipelineTS.metrics import pinaw
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = pinaw(yt, lower, upper)
        assert isinstance(result, float)
        assert result > 0

    def test_winkler_score(self):
        from PipelineTS.metrics import winkler_score
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = winkler_score(yt, lower, upper, alpha=0.1)
        assert isinstance(result, float)
        assert result == 1.0  # all within bounds, score = width

    def test_winkler_score_with_penalty(self):
        from PipelineTS.metrics import winkler_score
        yt = np.array([10.0])  # way outside [0, 1]
        lower = np.array([0.0])
        upper = np.array([1.0])
        result = winkler_score(yt, lower, upper, alpha=0.1)
        assert result > 1.0  # penalized


# ═══════════════════════════════════════════════════════════════════════════════
#  P1: PREPROCESSING — Stationarity, Frequency, Split
# ═══════════════════════════════════════════════════════════════════════════════

class TestStationarityTest:
    def test_adf_stationary(self):
        from PipelineTS.preprocessing import StationarityTest
        np.random.seed(42)
        stationary = np.random.randn(200)
        tester = StationarityTest()
        result = tester.adf_test(stationary)
        assert result['is_stationary'] == True

    def test_adf_nonstationary(self):
        from PipelineTS.preprocessing import StationarityTest
        nonstationary = np.cumsum(np.random.randn(200))
        tester = StationarityTest()
        result = tester.adf_test(nonstationary)
        assert result['is_stationary'] == False

    def test_combined_test(self):
        from PipelineTS.preprocessing import StationarityTest
        np.random.seed(42)
        series = np.random.randn(200)
        tester = StationarityTest()
        result = tester.fit(series)
        assert 'conclusion' in result
        assert 'suggested_action' in result
        assert result['conclusion'] in ['stationary', 'trend_stationary',
                                         'difference_stationary', 'non_stationary']

    def test_suggest_differencing(self):
        from PipelineTS.preprocessing import StationarityTest
        nonstationary = np.cumsum(np.random.randn(200))
        tester = StationarityTest()
        d = tester.suggest_differencing(nonstationary)
        assert d >= 1


class TestFrequencyDetector:
    def test_detect_daily(self):
        from PipelineTS.preprocessing.time_series_analysis import FrequencyDetector
        df = make_ts_data(100, freq='D')
        detector = FrequencyDetector(time_col='date')
        info = detector.fit(df)
        assert info['is_regular'] is True
        assert 'D' in info['freq'] or 'day' in info['freq'].lower()

    def test_detect_with_periods(self):
        from PipelineTS.preprocessing.time_series_analysis import FrequencyDetector
        df = make_ts_data(200, freq='D')
        detector = FrequencyDetector(time_col='date')
        info = detector.fit(df, target_col='value')
        assert 'dominant_periods' in info
        assert isinstance(info['dominant_periods'], list)


class TestTimeSeriesSplit:
    def test_simple_split(self):
        from PipelineTS.preprocessing import TimeSeriesSplit
        df = make_ts_data(100)
        train, test = TimeSeriesSplit.split(df, time_col='date', test_size=0.2)
        assert len(train) == 80
        assert len(test) == 20
        assert train['date'].max() < test['date'].min()

    def test_split_int_size(self):
        from PipelineTS.preprocessing import TimeSeriesSplit
        df = make_ts_data(100)
        train, test = TimeSeriesSplit.split(df, time_col='date', test_size=30)
        assert len(test) == 30

    def test_expanding_window(self):
        from PipelineTS.preprocessing import TimeSeriesSplit
        df = make_ts_data(100)
        folds = list(TimeSeriesSplit.expanding_window(
            df, time_col='date', min_train_size=50, test_size=10, step=10
        ))
        assert len(folds) >= 1
        for train, test in folds:
            assert len(test) == 10
            assert train['date'].max() < test['date'].min()

    def test_sliding_window(self):
        from PipelineTS.preprocessing import TimeSeriesSplit
        df = make_ts_data(100)
        folds = list(TimeSeriesSplit.sliding_window(
            df, time_col='date', train_size=40, test_size=10, step=10
        ))
        assert len(folds) >= 1
        for train, test in folds:
            assert len(train) == 40
            assert len(test) == 10


# ═══════════════════════════════════════════════════════════════════════════════
#  P1: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourierFeatures:
    def test_basic(self):
        from PipelineTS.feature_engineering import FourierFeatures
        df = make_ts_data(100)
        ff = FourierFeatures(time_col='date', periods=[7, 30], n_harmonics=2)
        result = ff.transform(df)
        assert 'fourier_7_sin_1' in result.columns
        assert 'fourier_30_cos_2' in result.columns
        assert len(result) == 100

    def test_named_periods(self):
        from PipelineTS.feature_engineering import FourierFeatures
        ff = FourierFeatures(time_col='date', periods={'weekly': 7, 'monthly': 30})
        df = make_ts_data(50)
        result = ff.transform(df)
        assert 'fourier_weekly_sin_1' in result.columns

    def test_get_feature_names(self):
        from PipelineTS.feature_engineering import FourierFeatures
        ff = FourierFeatures(time_col='date', periods=[7], n_harmonics=3)
        names = ff.get_feature_names()
        assert len(names) == 6  # 3 harmonics * 2 (sin + cos)


class TestHolidayFeatures:
    def test_generic_holidays(self):
        from PipelineTS.feature_engineering import HolidayFeatures
        df = make_ts_data(400)  # > 1 year to hit some holidays
        hf = HolidayFeatures(time_col='date')
        result = hf.transform(df)
        assert 'holiday_is_holiday' in result.columns
        assert 'holiday_days_to_nearest' in result.columns
        assert 'holiday_near_holiday' in result.columns
        assert result['holiday_is_holiday'].sum() > 0

    def test_custom_holidays(self):
        from PipelineTS.feature_engineering import HolidayFeatures
        df = make_ts_data(100)
        hf = HolidayFeatures(time_col='date', custom_holidays=['2020-01-15', '2020-02-20'])
        result = hf.transform(df)
        jan15 = result[result['date'] == '2020-01-15']
        assert len(jan15) == 1
        assert jan15['holiday_is_holiday'].values[0] == 1

    def test_chinese_calendar(self):
        """Test chinese-calendar integration for country='CN'."""
        from PipelineTS.feature_engineering import HolidayFeatures
        # Use a date range covering 2024 National Day
        dates = pd.date_range('2024-09-28', '2024-10-08', freq='D')
        df = pd.DataFrame({'date': dates, 'value': np.random.randn(len(dates))})
        hf = HolidayFeatures(time_col='date', country='CN')
        result = hf.transform(df)
        # Basic holiday columns
        assert 'holiday_is_holiday' in result.columns
        # China-specific columns
        assert 'holiday_is_workday' in result.columns
        assert 'holiday_is_in_lieu' in result.columns
        assert 'holiday_holiday_name' in result.columns
        # Oct 1 = National Day holiday
        oct1 = result[result['date'] == '2024-10-01']
        assert oct1['holiday_is_holiday'].values[0] == 1
        assert oct1['holiday_is_workday'].values[0] == 0
        assert oct1['holiday_holiday_name'].values[0] == 'National Day'
        # Oct 4 or 7 = in_lieu (调休)
        in_lieu_days = result[result['holiday_is_in_lieu'] == 1]
        assert len(in_lieu_days) > 0
        # Oct 8 = normal workday
        oct8 = result[result['date'] == '2024-10-08']
        assert oct8['holiday_is_workday'].values[0] == 1
        assert oct8['holiday_is_holiday'].values[0] == 0

    def test_chinese_calendar_feature_names(self):
        """Test get_feature_names includes CN-specific features."""
        from PipelineTS.feature_engineering import HolidayFeatures
        hf = HolidayFeatures(time_col='date', country='CN')
        names = hf.get_feature_names()
        assert 'holiday_is_workday' in names
        assert 'holiday_is_in_lieu' in names
        assert 'holiday_holiday_name' in names


class TestLagFeatures:
    def test_basic(self):
        from PipelineTS.feature_engineering import LagFeatureExtractor
        df = make_ts_data(100)
        extractor = LagFeatureExtractor(
            time_col='date', target_col='value', window=10,
            features=['mean', 'std', 'trend_slope']
        )
        result = extractor.transform(df)
        assert 'lag_mean' in result.columns
        assert 'lag_std' in result.columns
        assert 'lag_trend_slope' in result.columns
        # First (window-1) rows should be NaN
        assert result['lag_mean'].isna().sum() == 9

    def test_all_features(self):
        from PipelineTS.feature_engineering import LagFeatureExtractor
        df = make_ts_data(100)
        extractor = LagFeatureExtractor(
            time_col='date', target_col='value', window=10, features='all'
        )
        result = extractor.transform(df)
        assert len(extractor.get_feature_names()) == 15


class TestFeaturePipeline:
    def test_combined(self):
        from PipelineTS.feature_engineering import TimeSeriesFeatureEngineer
        df = make_ts_data(100)
        engineer = TimeSeriesFeatureEngineer(
            time_col='date',
            target_col='value',
            use_calendar=True,
            use_fourier=True,
            fourier_periods=[7, 30],
            use_lags=True,
            lag_window=10,
            lag_features=['mean', 'std'],
        )
        result = engineer.fit_transform(df)
        assert len(result.columns) > len(df.columns)
        assert 'lag_mean' in result.columns
        assert 'fourier_7_sin_1' in result.columns

    def test_repr(self):
        from PipelineTS.feature_engineering import TimeSeriesFeatureEngineer
        eng = TimeSeriesFeatureEngineer(time_col='date', use_calendar=True)
        assert 'calendar' in repr(eng)


# ═══════════════════════════════════════════════════════════════════════════════
#  P1: EVALUATION — Residual Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestResidualAnalyzer:
    def test_statistics(self):
        from PipelineTS.evaluation import ResidualAnalyzer
        np.random.seed(42)
        y_true = np.random.randn(100) + 10
        y_pred = y_true + np.random.randn(100) * 0.5
        analyzer = ResidualAnalyzer(y_true, y_pred)
        stats = analyzer.statistics()
        assert 'mean' in stats
        assert 'rmse' in stats
        assert stats['rmse'] > 0

    def test_normality_test(self):
        from PipelineTS.evaluation import ResidualAnalyzer
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        analyzer = ResidualAnalyzer(y_true, y_pred)
        norm = analyzer.normality_test()
        assert 'shapiro' in norm
        assert 'jarque_bera' in norm

    def test_autocorrelation(self):
        from PipelineTS.evaluation import ResidualAnalyzer
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        analyzer = ResidualAnalyzer(y_true, y_pred)
        acorr = analyzer.autocorrelation()
        assert 'acf_values' in acorr
        assert 'ljung_box' in acorr

    def test_bias_analysis(self):
        from PipelineTS.evaluation import ResidualAnalyzer
        y_true = np.ones(100)
        y_pred = np.ones(100) * 0.5  # consistent under-prediction
        analyzer = ResidualAnalyzer(y_true, y_pred)
        bias = analyzer.bias_analysis()
        assert bias['mean_bias'] > 0
        assert 'under-predicting' in bias['bias_direction']

    def test_print_report(self, capsys):
        from PipelineTS.evaluation import ResidualAnalyzer
        np.random.seed(42)
        analyzer = ResidualAnalyzer(np.random.randn(50), np.random.randn(50))
        analyzer.report()
        captured = capsys.readouterr()
        assert 'RESIDUAL ANALYSIS' in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
#  P1: EVALUATION — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelComparison:
    def test_evaluate(self):
        from PipelineTS.evaluation import ModelComparison
        from PipelineTS.metrics import mape, r2_score
        comp = ModelComparison(time_col='date', target_col='value')
        np.random.seed(42)
        yt = np.random.randn(50) + 100
        comp.add_result('ModelA', yt, yt + np.random.randn(50) * 2)
        comp.add_result('ModelB', yt, yt + np.random.randn(50) * 5)
        table = comp.fit(metrics={'MAPE': mape, 'R2': r2_score})
        assert len(table) == 2
        assert 'MAPE' in table.columns
        assert 'R2' in table.columns

    def test_rank(self):
        from PipelineTS.evaluation import ModelComparison
        comp = ModelComparison(time_col='date', target_col='value')
        yt = np.array([1.0, 2.0, 3.0])
        comp.add_result('Good', yt, yt + 0.1)
        comp.add_result('Bad', yt, yt + 10.0)
        comp.fit()
        ranked = comp.rank('MAE', ascending=True)
        assert ranked.iloc[0]['model'] == 'Good'

    def test_with_interval_metrics(self):
        from PipelineTS.evaluation import ModelComparison
        from PipelineTS.metrics import picp, pinaw
        comp = ModelComparison(time_col='date', target_col='value')
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        comp.add_result('M1', yt, yt + 0.1,
                        lower=yt - 0.5, upper=yt + 0.5)
        table = comp.fit(
            interval_metrics={'PICP': picp, 'PINAW': pinaw}
        )
        assert 'PICP' in table.columns
        assert table.iloc[0]['PICP'] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  P2: TRAINING — Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightedEnsemble:
    def test_manual_weights(self):
        from PipelineTS.training import WeightedEnsemble
        from PipelineTS.ml_model import LightGBMModel
        data = make_ts_data(120)
        m1 = LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)
        m2 = LightGBMModel(time_col='date', target_col='value', lags=12,
                           n_estimators=50, verbose=-1)
        ens = WeightedEnsemble(
            [('lgbm1', m1), ('lgbm2', m2)],
            time_col='date', target_col='value',
            weights=[0.6, 0.4],
        )
        ens.fit(data)
        result = ens.predict(5)
        assert len(result) == 5
        assert 'value' in result.columns

    def test_auto_weights(self):
        from PipelineTS.training import WeightedEnsemble
        from PipelineTS.ml_model import LightGBMModel
        data = make_ts_data(120)
        m1 = LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)
        m2 = LightGBMModel(time_col='date', target_col='value', lags=12,
                           n_estimators=50, verbose=-1)
        ens = WeightedEnsemble(
            [('lgbm1', m1), ('lgbm2', m2)],
            time_col='date', target_col='value',
            weights='auto',
        )
        ens.fit(data)
        weights = ens.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        result = ens.predict(5)
        assert len(result) == 5


# ═══════════════════════════════════════════════════════════════════════════════
#  P2: PREDICTION — Rolling Predict
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollingPredictor:
    def test_basic_rolling(self):
        from PipelineTS.prediction import RollingPredictor
        from PipelineTS.ml_model import LightGBMModel
        data = make_ts_data(150)
        model = LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)
        rp = RollingPredictor(
            model, time_col='date', target_col='value',
            train_size=80, horizon=10, step=20, refit=True,
        )
        results = rp.predict(data, verbose=False)
        assert len(results) > 0
        assert 'value' in results.columns
        assert 'value_actual' in results.columns
        assert 'window_id' in results.columns

    def test_evaluate(self):
        from PipelineTS.prediction import RollingPredictor
        from PipelineTS.ml_model import LightGBMModel
        data = make_ts_data(150)
        model = LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)
        rp = RollingPredictor(
            model, time_col='date', target_col='value',
            train_size=80, horizon=10, step=20,
        )
        results = rp.predict(data, verbose=False)
        eval_results = rp.score(results)
        assert 'MAE' in eval_results
        assert eval_results['MAE']['overall'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  P2: PREDICTION — Explainability
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelExplainer:
    def test_feature_importance_gbdt(self):
        from PipelineTS.prediction import ModelExplainer
        from PipelineTS.ml_model import LightGBMModel
        data = make_ts_data(120)
        model = LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)
        model.fit(data)
        explainer = ModelExplainer(model, time_col='date', target_col='value')
        importance = explainer.feature_importance()
        assert importance is not None
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
