import argparse
from pathlib import Path

from PipelineTS.dataset import BuiltInSeriesData
from PipelineTS.pipeline import SmartRouter


DATASET_ALIASES = {
    "air_passengers": "AirPassengers",
    "airpassengers": "AirPassengers",
    "electric": "Electric_Production",
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "messages": "Messages_Sent",
    "messages_hour": "Messages_Sent_Hour",
    "supermarket": "Supermarket_Incoming",
    "web_sales": "Web_Sales",
}


FIELDS = [
    "n_rows",
    "freq",
    "is_regular",
    "stationarity",
    "trend_strength",
    "seasonality_strength",
    "noise_ratio",
    "autocorr_lag1",
    "n_seasonalities",
    "regime_changes",
    "pct_outlier",
    "cv",
    "kurtosis",
]


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def dataset_alias(name):
    aliases = {
        "AirPassengers": "air_passengers",
        "Electric_Production": "electric",
        "Messages_Sent": "messages",
        "Messages_Sent_Hour": "messages_hour",
        "Supermarket_Incoming": "supermarket",
        "Web_Sales": "web_sales",
    }
    return aliases.get(name, name.lower())


def discover_builtin_datasets():
    source = BuiltInSeriesData(print_file_list=False)
    datasets = {}
    for filename in source.file_list:
        if not filename.endswith(".csv"):
            continue
        name = Path(filename).stem
        wrapper = source[name]
        datasets[name] = {
            "name": name,
            "alias": dataset_alias(name),
            "time_col": wrapper.time_col,
            "target_col": wrapper.target_col,
        }
    return datasets


def resolve_dataset_names(names):
    datasets = discover_builtin_datasets()
    lookup = {}
    for name, spec in datasets.items():
        keys = {
            name,
            name.lower(),
            spec["alias"],
            spec["alias"].lower(),
            name.lower().replace("_", ""),
            spec["alias"].lower().replace("_", ""),
        }
        for key in keys:
            lookup[key] = name
    for alias, canonical in DATASET_ALIASES.items():
        if canonical in datasets:
            lookup[alias] = canonical

    if not names or any(item.lower() == "all" for item in names):
        return [datasets[name] for name in sorted(datasets)]

    resolved = []
    for item in names:
        key = item.lower()
        canonical = lookup.get(key)
        if canonical is None:
            choices = ", ".join(spec["alias"] for spec in resolve_dataset_names(["all"]))
            raise ValueError(f"Unknown dataset: {item}. Available: {choices}")
        resolved.append(datasets[canonical])
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--n-tail", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()

    source = BuiltInSeriesData(print_file_list=False)
    dataset_specs = resolve_dataset_names(parse_csv(args.datasets))
    print("dataset " + " ".join(f"{field:>20}" for field in FIELDS))
    for spec in dataset_specs:
        time_col = spec["time_col"]
        target_col = spec["target_col"]
        data = source[spec["name"]][[time_col, target_col]].dropna()
        if args.n_tail and len(data) > args.n_tail:
            data = data.tail(args.n_tail)
        data = data.reset_index(drop=True)
        router = SmartRouter(time_col=time_col, target_col=target_col, n_predict=args.horizon, verbose=False)
        data = router._ensure_datetime(data)
        profile = router._profile_data(data)
        values = []
        for field in FIELDS:
            value = getattr(profile, field)
            if isinstance(value, float):
                values.append(f"{value:>20.4f}")
            else:
                values.append(f"{str(value):>20}")
        print(f"{spec['alias']:<16}" + "".join(values))


if __name__ == "__main__":
    main()
