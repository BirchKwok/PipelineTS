import pandas as pd

from PipelineTS.dataset import LoadElectric
from PipelineTS.pipeline import ModelPipeline, SmartRouter


def main():
    data = LoadElectricProduction()[["date", "value"]].dropna().tail(120).reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    train = data.iloc[:-12].reset_index(drop=True)
    valid = data.iloc[-12:].reset_index(drop=True)
    models = ["naive", "seasonal_naive", "theta", "ets", "stat_ensemble"]
    pipe = ModelPipeline(
        time_col="date",
        target_col="value",
        lags=12,
        include_models=models,
        quantile=None,
        scaler=True,
        cv=2,
        time_limit=30,
    )
    lb = pipe.fit(train, valid_data=valid)
    pred = pipe.predict(12)
    available = set(ModelPipeline.list_all_available_models())
    router = SmartRouter(time_col="date", target_col="value", n_predict=12, preset="fast", verbose=False)
    selected = router._baseline_guardrail_candidates(12) if router.profile_ is not None else []
    assert all(m in available for m in models)
    assert not lb.empty
    assert len(pred) == 12
    print("PASS stat_baselines best={} metric={:.4f} rows={}".format(lb.iloc[0]["model"], float(lb.iloc[0]["metric"]), len(pred)))


if __name__ == "__main__":
    main()
