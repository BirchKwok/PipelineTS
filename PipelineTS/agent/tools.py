"""Tool schema definitions in OpenAI function-calling format.

Each tool is a dict with 'type': 'function', 'function': {name, description, parameters}.
"""

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

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
    # Preprocessing
    FILL_MISSING_VALUES,
    HANDLE_OUTLIERS,
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
