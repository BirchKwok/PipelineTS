"""Tool schema definitions in OpenAI function-calling format.

Each tool is a dict with 'type': 'function', 'function': {name, description, parameters}.
"""

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }

LOAD_CSV = {
    "type": "function",
    "function": {
        "name": "load_csv",
        "description": "Load time series data from a CSV file into memory. Must be called before any analysis or training. When data was uploaded via the web UI, use the filepath shown in the session state.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the CSV file (absolute or relative to current directory).",
                },
                "time_col": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Name of the datetime column in the CSV (e.g., 'date', 'timestamp'). If multiple are provided, the first is used as the primary time column.",
                },
                "target_col": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Name of the target/value column(s) to forecast (e.g., 'value', 'sales'). Use an array for multi-target forecasting.",
                },
                "id_col": {
                    "type": "string",
                    "description": "Optional: name of the series identifier column for multi-series (panel) data.",
                },
                "sep": {
                    "type": "string",
                    "description": "CSV delimiter (default: ',').",
                },
            },
            "required": ["filepath", "time_col", "target_col"],
        },
    },
}

LOAD_BUILTIN_DATASET = {
    "type": "function",
    "function": {
        "name": "load_builtin_dataset",
        "description": "Load one of the built-in example time series datasets for quick experimentation.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "enum": [
                        "electric",
                        "messages_sent_hour",
                        "messages_sent",
                        "web_sales",
                        "supermarket_incoming",
                    ],
                    "description": "Name of the built-in dataset to load.",
                },
            },
            "required": ["dataset_name"],
        },
    },
}

# ---------------------------------------------------------------------------
# Data Inspection
# ---------------------------------------------------------------------------

INSPECT_DATA = {
    "type": "function",
    "function": {
        "name": "inspect_data",
        "description": "Inspect the currently loaded dataset: show shape, dtypes, first/last rows, and summary statistics.",
        "parameters": {
            "type": "object",
            "properties": {
                "n_rows": {
                    "type": "integer",
                    "description": "Number of rows to show from head and tail (default: 5).",
                },
            },
            "required": [],
        },
    },
}

CHECK_MISSING_VALUES = {
    "type": "function",
    "function": {
        "name": "check_missing_values",
        "description": "Detect missing values in the time series — both explicit NaN values and implicit gaps in the time index.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

DETECT_OUTLIERS = {
    "type": "function",
    "function": {
        "name": "detect_outliers",
        "description": "Detect outliers in the target variable using the IQR or Z-score method.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["iqr", "zscore"],
                    "description": "Detection method: 'iqr' (interquartile range) or 'zscore' (Z-score). Default: 'iqr'.",
                },
            },
            "required": [],
        },
    },
}

CHECK_STATIONARITY = {
    "type": "function",
    "function": {
        "name": "check_stationarity",
        "description": "Test whether the time series is stationary using ADF and KPSS tests. Suggests differencing order if needed.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

DATA_QUALITY_REPORT = {
    "type": "function",
    "function": {
        "name": "data_quality_report",
        "description": "Generate a comprehensive data quality report including missing values, outliers, frequency, and basic statistics.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

GET_DATA_CONTEXT = {
    "type": "function",
    "function": {
        "name": "get_data_context",
        "description": "Return evidence-backed numeric context from the actual dataset for selected rows, same-day/all-day rows, the full dataset, or comparisons between them. Use this when the user asks how a confirmed selection differs from the whole day, overall data, full dataset, surrounding period, or a specific column such as HUFL. This tool is intentionally not auto-limited to the confirmed selection.",
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": [
                        "selected",
                        "same_day",
                        "full_dataset",
                        "selected_vs_same_day",
                        "selected_vs_full_dataset",
                    ],
                    "description": "Data scope to summarize. Use 'selected_vs_same_day' for questions involving 全天/all-day context around the confirmed selection. Default: selected_vs_same_day when a selection exists, otherwise full_dataset.",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Numeric columns to summarize. If omitted, uses selected non-time columns, then the current target column, then available numeric columns.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to include in each preview table. Default: 40.",
                },
                "include_preview": {
                    "type": "boolean",
                    "description": "Whether to include row previews for the returned scopes. Default: true.",
                },
            },
            "required": [],
        },
    },
}

ANALYZE_TIME_INDEX = _tool(
    "analyze_time_index",
    "Analyze timestamp quality: validity, monotonicity, duplicates, inferred frequency, interval distribution, irregular gaps, and large gap examples.",
)

PROFILE_SERIES = _tool(
    "profile_series",
    "Compute rich target-series characteristics: quantiles, skew/kurtosis, coefficient of variation, zeros/negatives, Hurst exponent, spectral entropy, and transformation hints.",
)

ANALYZE_AUTOCORRELATION = _tool(
    "analyze_autocorrelation",
    "Analyze ACF/PACF and Ljung-Box autocorrelation to identify memory, AR structure, and useful lag candidates.",
    {
        "max_lags": {
            "type": "integer",
            "description": "Maximum lag to analyze. Default: 40.",
        },
    },
)

DETECT_SEASONALITY = _tool(
    "detect_seasonality",
    "Detect seasonal periods using FFT spectral peaks, ACF peaks, and optional STL seasonal-strength analysis.",
    {
        "period": {
            "type": "integer",
            "description": "Optional known seasonal period to evaluate with STL.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of candidate periods to report. Default: 5.",
        },
    },
)

ANALYZE_TREND = _tool(
    "analyze_trend",
    "Analyze trend direction and strength using linear slope, Kendall trend test, rolling slope, and trend sign reversals.",
    {
        "window": {
            "type": "integer",
            "description": "Rolling window for local trend slopes. Auto-selected if omitted.",
        },
    },
)

DETECT_CHANGEPOINTS = _tool(
    "detect_changepoints",
    "Detect likely structural breaks/changepoints in mean, variance, or CUSUM-style regime changes.",
    {
        "method": {
            "type": "string",
            "enum": ["auto", "mean", "variance", "cusum"],
            "description": "Changepoint scoring method. Default: auto.",
        },
        "window": {
            "type": "integer",
            "description": "Comparison window around each candidate point. Auto-selected if omitted.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of changepoints to report. Default: 5.",
        },
    },
)

DETECT_DISTRIBUTION_SHIFT = _tool(
    "detect_distribution_shift",
    "Compare early/middle/recent segments to detect distribution drift using segment statistics and Kolmogorov-Smirnov tests.",
    {
        "segments": {
            "type": "integer",
            "description": "Number of chronological segments to compare. Default: 3.",
        },
    },
)

ANALYZE_VOLATILITY = _tool(
    "analyze_volatility",
    "Analyze rolling volatility, coefficient of variation, volatility trend, high-volatility windows, and volatility clustering.",
    {
        "window": {
            "type": "integer",
            "description": "Rolling window for volatility statistics. Auto-selected if omitted.",
        },
    },
)

SUGGEST_LAG_FEATURES = _tool(
    "suggest_lag_features",
    "Suggest useful lag windows and lag features based on autocorrelation significance and top lag correlations.",
    {
        "max_lags": {
            "type": "integer",
            "description": "Maximum lag to consider. Default: 60.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of top lag correlations to report. Default: 10.",
        },
    },
)

DETECT_CALENDAR_EFFECTS = _tool(
    "detect_calendar_effects",
    "Detect calendar effects by comparing target averages across hour, weekday, day-of-month, month, and quarter groups.",
    {
        "granularity": {
            "type": "string",
            "enum": ["auto", "hour", "weekday", "dayofmonth", "month", "quarter"],
            "description": "Calendar grouping to analyze. Default: auto analyzes all available groupings.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of strongest groups to show. Default: 10.",
        },
    },
)

ANALYZE_COVARIATES = _tool(
    "analyze_covariates",
    "Analyze numeric covariate relationships with the target using Pearson/Spearman correlations and lead/lag correlations.",
    {
        "covariates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of covariate columns. If omitted, all numeric non-target columns are considered.",
        },
        "max_lag": {
            "type": "integer",
            "description": "Maximum positive/negative lag to test. Default: 12.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of strongest covariates to report. Default: 10.",
        },
    },
)

ANALYZE_INTERMITTENCY = _tool(
    "analyze_intermittency",
    "Classify demand pattern as smooth/intermittent/erratic/lumpy using ADI and CV², with zero-ratio and nonzero-gap statistics.",
)

DECOMPOSE_COMPONENTS = _tool(
    "decompose_components",
    "Summarize STL component strengths for trend, seasonality, and residual noise without generating a plot.",
    {
        "period": {
            "type": "integer",
            "description": "Seasonal period for STL. Auto-detected if omitted.",
        },
    },
)

RECOMMEND_TIMESERIES_ACTIONS = _tool(
    "recommend_timeseries_actions",
    "Generate actionable recommendations for preprocessing, transformations, lag features, seasonal features, and model families based on diagnostics.",
)

ASSESS_FORECASTABILITY = _tool(
    "assess_forecastability",
    "Assess whether the target series is intrinsically forecastable using memory, entropy, seasonal/trend strengths, history length, and noise indicators.",
    {
        "horizon": {
            "type": "integer",
            "description": "Forecast horizon used to judge whether history is sufficient.",
        },
        "seasonal_period": {
            "type": "integer",
            "description": "Optional known seasonal period to evaluate. Auto-detected if omitted.",
        },
    },
)

BENCHMARK_BASELINES = _tool(
    "benchmark_baselines",
    "Benchmark simple naive forecasting baselines on a holdout window so trained models have a concrete performance target.",
    {
        "horizon": {
            "type": "integer",
            "description": "Forecast horizon to use as holdout size when test_size is omitted.",
        },
        "seasonal_period": {
            "type": "integer",
            "description": "Optional seasonal period for seasonal naive baseline. Auto-detected if omitted.",
        },
        "test_size": {
            "type": "integer",
            "description": "Explicit holdout size for baseline evaluation.",
        },
    },
)

ANALYZE_PANEL_STRUCTURE = _tool(
    "analyze_panel_structure",
    "Analyze multi-series/panel structure: series count, length balance, duplicate id-time keys, per-series regularity, coverage, and target heterogeneity.",
)

DETECT_LEAKAGE_RISK = _tool(
    "detect_leakage_risk",
    "Detect likely target leakage or invalid covariate usage from feature names, same-time target correlation, lead correlation, and configured known/past covariates.",
    {
        "horizon": {
            "type": "integer",
            "description": "Maximum lead horizon to inspect for future-target leakage. Default: 12.",
        },
        "corr_threshold": {
            "type": "number",
            "description": "Correlation threshold for review findings. Default: 0.98.",
        },
    },
)

ASSESS_MODELING_READINESS = _tool(
    "assess_modeling_readiness",
    "Assess whether the dataset is ready for modeling, including blocking issues, warnings, validation horizon guidance, lag/seasonality hints, covariate availability, and panel concerns.",
    {
        "horizon": {
            "type": "integer",
            "description": "Forecast horizon used for readiness and validation recommendations.",
        },
    },
)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

FILL_MISSING_VALUES = {
    "type": "function",
    "function": {
        "name": "fill_missing_values",
        "description": "Fill missing values in the time series using the specified interpolation method.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["linear", "ffill", "bfill", "spline", "zero"],
                    "description": "Interpolation method: 'linear' (default), 'ffill' (forward fill), 'bfill' (backward fill), 'spline', or 'zero'.",
                },
            },
            "required": [],
        },
    },
}

HANDLE_OUTLIERS = {
    "type": "function",
    "function": {
        "name": "handle_outliers",
        "description": "Handle detected outliers using the specified strategy.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["clip", "nan", "median", "linear"],
                    "description": "Strategy: 'clip' (cap at bounds), 'nan' (set to NaN then interpolate), 'median' (replace with median), 'linear' (linear interpolation). Default: 'clip'.",
                },
            },
            "required": [],
        },
    },
}

SORT_AND_DEDUPLICATE = _tool(
    "sort_and_deduplicate",
    "Sort data by time (and id column for panel data), remove invalid timestamps, and aggregate duplicate timestamps.",
    {
        "duplicate_strategy": {
            "type": "string",
            "enum": ["mean", "sum", "median", "min", "max", "first", "last"],
            "description": "Aggregation strategy for numeric duplicate timestamp rows. Default: mean.",
        },
    },
)

RESAMPLE_TIME_SERIES = _tool(
    "resample_time_series",
    "Regularize or resample the time series to a fixed frequency, aggregating numeric columns and filling gaps.",
    {
        "freq": {
            "type": "string",
            "description": "Target pandas frequency string such as 'D', 'H', 'W', 'MS'. Auto-inferred if omitted.",
        },
        "agg": {
            "type": "string",
            "enum": ["mean", "sum", "median", "min", "max", "first", "last"],
            "description": "Aggregation for numeric columns. Default: mean.",
        },
        "fill_method": {
            "type": "string",
            "enum": ["linear", "ffill", "bfill", "zero", "none"],
            "description": "How to fill numeric gaps after resampling. Default: linear.",
        },
    },
)

TRANSFORM_TARGET = _tool(
    "transform_target",
    "Create transformed target column(s) or replace existing target values using log1p, sqrt, Box-Cox, Yeo-Johnson, standardize, or minmax.",
    {
        "method": {
            "type": "string",
            "enum": ["log1p", "sqrt", "boxcox", "yeojohnson", "standardize", "minmax"],
            "description": "Target transformation method.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to transform. Defaults to current target column(s).",
        },
        "suffix": {
            "type": "string",
            "description": "Suffix for created column(s). Defaults to method name.",
        },
        "replace": {
            "type": "boolean",
            "description": "If true, replace original column instead of creating a new one. Default: false.",
        },
    },
    required=["method"],
)

DIFFERENCE_SERIES = _tool(
    "difference_series",
    "Create differenced target columns for detrending or seasonal differencing.",
    {
        "order": {
            "type": "integer",
            "description": "Non-seasonal differencing order. Default: 1.",
        },
        "seasonal_period": {
            "type": "integer",
            "description": "Optional seasonal differencing period.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to difference. Defaults to current target column(s).",
        },
        "suffix": {
            "type": "string",
            "description": "Suffix for created columns. Defaults to diff order.",
        },
        "drop_na": {
            "type": "boolean",
            "description": "Drop initial rows with NaN after differencing. Default: false.",
        },
    },
)

SMOOTH_SERIES = _tool(
    "smooth_series",
    "Create rolling/EMA-smoothed target columns for trend extraction or noise reduction.",
    {
        "method": {
            "type": "string",
            "enum": ["rolling_mean", "rolling_median", "ewm"],
            "description": "Smoothing method. Default: rolling_mean.",
        },
        "window": {
            "type": "integer",
            "description": "Rolling window or EWM span. Default: 7.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to smooth. Defaults to current target column(s).",
        },
        "suffix": {
            "type": "string",
            "description": "Suffix for created columns.",
        },
        "replace": {
            "type": "boolean",
            "description": "If true, replace original column instead of creating a new one. Default: false.",
        },
    },
)

CLIP_OR_WINSORIZE = _tool(
    "clip_or_winsorize",
    "Clip numeric target columns to lower/upper quantiles for robust outlier handling.",
    {
        "lower_q": {
            "type": "number",
            "description": "Lower quantile. Default: 0.01.",
        },
        "upper_q": {
            "type": "number",
            "description": "Upper quantile. Default: 0.99.",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to clip. Defaults to current target column(s).",
        },
        "replace": {
            "type": "boolean",
            "description": "If true, replace original columns. Default: true.",
        },
        "suffix": {
            "type": "string",
            "description": "Suffix when replace=false. Default: winsor.",
        },
    },
)

SET_COVARIATES = _tool(
    "set_covariates",
    "Configure known future covariates, past covariates, and general feature columns for later model training.",
    {
        "known_covariates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns known into the forecast horizon, such as holidays, prices, promotions, planned capacity.",
        },
        "past_covariates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns only observed historically, such as weather measurements or sensor readings.",
        },
        "feature_cols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "General feature columns for multivariate models.",
        },
    },
)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PLOT_TIME_SERIES = {
    "type": "function",
    "function": {
        "name": "plot_time_series",
        "description": "Plot the raw time series data. The plot is saved to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "save_path": {
                    "type": "string",
                    "description": "File path to save the plot image (PNG, PDF, etc.). Default: 'plot_series.png'.",
                },
                "title": {
                    "type": "string",
                    "description": "Optional plot title.",
                },
            },
            "required": [],
        },
    },
}

PLOT_ACF_PACF = {
    "type": "function",
    "function": {
        "name": "plot_acf_pacf",
        "description": "Plot ACF (autocorrelation) and PACF (partial autocorrelation) side by side to help determine AR/MA orders.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_lags": {
                    "type": "integer",
                    "description": "Number of lags to include (default: 30).",
                },
                "save_path": {
                    "type": "string",
                    "description": "File path to save the plot. Default: 'plot_acf_pacf.png'.",
                },
            },
            "required": [],
        },
    },
}

PLOT_DECOMPOSITION = {
    "type": "function",
    "function": {
        "name": "plot_decomposition",
        "description": "Decompose the time series into trend, seasonal, and residual components and plot them.",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "integer",
                    "description": "Seasonal period for decomposition (e.g., 12 for monthly data with yearly seasonality). Auto-detected if not provided.",
                },
                "save_path": {
                    "type": "string",
                    "description": "File path to save the plot. Default: 'plot_decomposition.png'.",
                },
            },
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

CREATE_FEATURES = {
    "type": "function",
    "function": {
        "name": "create_features",
        "description": "Create time series features: lag features, Fourier features, calendar features, and holiday features.",
        "parameters": {
            "type": "object",
            "properties": {
                "use_lags": {
                    "type": "boolean",
                    "description": "Whether to create rolling lag features (default: false).",
                },
                "lag_window": {
                    "type": "integer",
                    "description": "Window size for lag features (default: 12).",
                },
                "use_fourier": {
                    "type": "boolean",
                    "description": "Whether to create Fourier periodic features (default: false).",
                },
                "fourier_periods": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Periods for Fourier features (e.g., [7, 365] for weekly and yearly).",
                },
                "use_calendar": {
                    "type": "boolean",
                    "description": "Whether to create calendar features like weekday, month, etc. (default: false).",
                },
                "use_holidays": {
                    "type": "boolean",
                    "description": "Whether to create holiday indicator features (default: false).",
                },
                "holiday_country": {
                    "type": "string",
                    "description": "Country code for holidays: 'US' or 'CN' (default: 'US').",
                },
            },
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------

LIST_AVAILABLE_MODELS = {
    "type": "function",
    "function": {
        "name": "list_available_models",
        "description": "List all available time series forecasting models in PipelineTS, grouped by category (NN, ML, statistical, foundation).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TRAIN_PIPELINE = {
    "type": "function",
    "function": {
        "name": "train_pipeline",
        "description": "Train a ModelPipeline: trains multiple models and automatically selects the best one based on validation performance.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_models": {
                    "type": "string",
                    "enum": ["light", "all", "nn", "ml"],
                    "description": "Model category to train: 'light' (fast, 11 models), 'all' (all models), 'nn' (neural networks only), 'ml' (machine learning only). Default: 'light'.",
                },
                "lags": {
                    "type": "integer",
                    "description": "Number of lagged time steps (input window size). Default: auto-determined.",
                },
                "quantile": {
                    "type": "number",
                    "description": "Coverage level for prediction intervals (e.g., 0.9 for 90% intervals). None = point forecasts only.",
                },
                "cv": {
                    "type": "integer",
                    "description": "Cross-validation folds (default: 5).",
                },
                "use_scaler": {
                    "type": "boolean",
                    "description": "Whether to scale data (default: true, uses MinMaxScaler).",
                },
            },
            "required": [],
        },
    },
}

TRAIN_SMART_ROUTER = {
    "type": "function",
    "function": {
        "name": "train_smart_router",
        "description": "Train a SmartRouter: intelligently analyzes data characteristics and automatically selects the best preprocessing, models, lags, and hyperparameters. Supports weighted ensemble of top models.",
        "parameters": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": ["fast", "medium_quality", "high_quality", "best_quality"],
                    "description": "Quality preset. 'fast' (~3 models, no ensemble), 'medium_quality' (5 models, auto ensemble), 'high_quality' (8 models, weighted ensemble), 'best_quality' (15 models, top-5 ensemble). Default: 'medium_quality'.",
                },
                "n_predict": {
                    "type": "integer",
                    "description": "Forecast horizon — number of future steps to predict.",
                },
                "quantile": {
                    "type": "number",
                    "description": "Coverage level for prediction intervals (e.g., 0.9).",
                },
                "time_limit": {
                    "type": "integer",
                    "description": "Total time budget in seconds. None = no limit.",
                },
                "include_models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: pin specific model(s) for SmartRouter to optimize. If not set, SmartRouter selects models automatically.",
                },
            },
            "required": ["n_predict"],
        },
    },
}

TRAIN_SINGLE_MODEL = {
    "type": "function",
    "function": {
        "name": "train_single_model",
        "description": "Train a single specific model. Use list_available_models to see available model keys.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model key name (e.g., 'torch_boosting_forest', 'd_linear', 'prophet', 'auto_arima'). Use list_available_models to see all options.",
                },
                "lags": {
                    "type": "integer",
                    "description": "Number of lagged time steps (default: auto-determined).",
                },
                "quantile": {
                    "type": "number",
                    "description": "Prediction interval coverage (e.g., 0.9).",
                },
            },
            "required": ["model_name"],
        },
    },
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

SHOW_LEADERBOARD = {
    "type": "function",
    "function": {
        "name": "show_leaderboard",
        "description": "Show the model leaderboard — models ranked by validation metric. Only available after training a ModelPipeline or SmartRouter.",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Show top N models (default: all).",
                },
            },
            "required": [],
        },
    },
}

BACKTEST_MODEL = {
    "type": "function",
    "function": {
        "name": "backtest_model",
        "description": "Run walk-forward backtesting to evaluate model performance over multiple forecast origins.",
        "parameters": {
            "type": "object",
            "properties": {
                "n_splits": {
                    "type": "integer",
                    "description": "Number of backtesting splits (default: 5).",
                },
                "test_size": {
                    "type": "integer",
                    "description": "Number of steps in each test window.",
                },
                "model_name": {
                    "type": "string",
                    "description": "Which model to backtest. If not specified, uses the best model.",
                },
            },
            "required": ["test_size"],
        },
    },
}

ANALYZE_RESIDUALS = {
    "type": "function",
    "function": {
        "name": "analyze_residuals",
        "description": "Analyze prediction residuals: statistics, normality tests, autocorrelation check, and bias analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model to analyze residuals for. If not specified, uses the best model.",
                },
            },
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

FORECAST = {
    "type": "function",
    "function": {
        "name": "forecast",
        "description": "Generate forecasts for the specified number of future steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of future steps to forecast.",
                },
                "model_name": {
                    "type": "string",
                    "description": "Model to use for forecasting. If not specified, uses the best model.",
                },
            },
            "required": ["n"],
        },
    },
}

PREDICT_WITH_INTERVALS = {
    "type": "function",
    "function": {
        "name": "predict_with_intervals",
        "description": "Generate forecasts with prediction intervals at multiple coverage levels (e.g., 50%, 80%, 95%).",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of future steps to forecast.",
                },
                "levels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Coverage levels for prediction intervals (e.g., [0.5, 0.8, 0.95]). Default: [0.5, 0.8, 0.9, 0.95].",
                },
                "model_name": {
                    "type": "string",
                    "description": "Model to use. If not specified, uses the best model.",
                },
            },
            "required": ["n"],
        },
    },
}

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

SAVE_MODEL = {
    "type": "function",
    "function": {
        "name": "save_model",
        "description": "Save the current model, pipeline, or SmartRouter to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "File path to save to (e.g., 'my_model.pts').",
                },
            },
            "required": ["filepath"],
        },
    },
}

LOAD_MODEL = {
    "type": "function",
    "function": {
        "name": "load_model",
        "description": "Load a previously saved model, pipeline, or SmartRouter from a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "File path to load from (e.g., 'my_model.pts').",
                },
            },
            "required": ["filepath"],
        },
    },
}

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

GET_SESSION_STATUS = {
    "type": "function",
    "function": {
        "name": "get_session_status",
        "description": "Get the current session status: what data is loaded, what models are trained, and current configuration.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# All tools list
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    # Data loading
    LOAD_CSV,
    LOAD_BUILTIN_DATASET,
    # Data inspection
    INSPECT_DATA,
    CHECK_MISSING_VALUES,
    DETECT_OUTLIERS,
    CHECK_STATIONARITY,
    DATA_QUALITY_REPORT,
    GET_DATA_CONTEXT,
    ANALYZE_TIME_INDEX,
    PROFILE_SERIES,
    ANALYZE_AUTOCORRELATION,
    DETECT_SEASONALITY,
    ANALYZE_TREND,
    DETECT_CHANGEPOINTS,
    DETECT_DISTRIBUTION_SHIFT,
    ANALYZE_VOLATILITY,
    SUGGEST_LAG_FEATURES,
    DETECT_CALENDAR_EFFECTS,
    ANALYZE_COVARIATES,
    ANALYZE_INTERMITTENCY,
    DECOMPOSE_COMPONENTS,
    RECOMMEND_TIMESERIES_ACTIONS,
    ASSESS_FORECASTABILITY,
    BENCHMARK_BASELINES,
    ANALYZE_PANEL_STRUCTURE,
    DETECT_LEAKAGE_RISK,
    ASSESS_MODELING_READINESS,
    # Preprocessing
    FILL_MISSING_VALUES,
    HANDLE_OUTLIERS,
    SORT_AND_DEDUPLICATE,
    RESAMPLE_TIME_SERIES,
    TRANSFORM_TARGET,
    DIFFERENCE_SERIES,
    SMOOTH_SERIES,
    CLIP_OR_WINSORIZE,
    SET_COVARIATES,
    # Visualization
    PLOT_TIME_SERIES,
    PLOT_ACF_PACF,
    PLOT_DECOMPOSITION,
    # Feature engineering
    CREATE_FEATURES,
    # Model management
    LIST_AVAILABLE_MODELS,
    TRAIN_PIPELINE,
    TRAIN_SMART_ROUTER,
    TRAIN_SINGLE_MODEL,
    # Evaluation
    SHOW_LEADERBOARD,
    BACKTEST_MODEL,
    ANALYZE_RESIDUALS,
    # Prediction
    FORECAST,
    PREDICT_WITH_INTERVALS,
    # Persistence
    SAVE_MODEL,
    LOAD_MODEL,
    # Session
    GET_SESSION_STATUS,
]
