import pandas as pd

from PipelineTS.dataset import LoadElectric
from PipelineTS.pipeline import SmartRouter


def main():
    data = LoadElectricProduction()[["date", "value"]].dropna().tail(180).reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    train = data.iloc[:-12].reset_index(drop=True)
    valid = data.iloc[-12:].reset_index(drop=True)
    router = SmartRouter(
        time_col="date",
        target_col="value",
        n_predict=12,
        preset="high_quality",
        time_limit=45,
        search_strategy="thorough",
        ensemble_strategy="weighted_avg",
        hpo_strategy="none",
        cv=2,
        verbose=False,
    )
    router.fit(train, valid_data=valid)
    assert router.leader_board_ is not None and not router.leader_board_.empty
    guardrail = router.strategy_.get("guardrails", {}).get("baseline", {})
    assert getattr(router, "_baseline_guardrail_cache", None) is not None
    assert guardrail.get("checked") is True
    print("PASS smartrouter_preflight best={} metric={:.4f} guardrail={}".format(
        router.leader_board_.iloc[0]["model"],
        float(router.leader_board_.iloc[0]["metric"]),
        guardrail.get("reason"),
    ))


if __name__ == "__main__":
    main()
