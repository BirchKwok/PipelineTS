import inspect

from PipelineTS.nn_model import (
    DLinearModel,
    DeepARModel,
    GAUModel,
    ITransformerModel,
    NBeatsModel,
    NHitsModel,
    NLinearModel,
    PatchRNNModel,
    SRSNetModel,
    StackingRNNModel,
    TCNModel,
    TFTModel,
    TiDEModel,
    Time2VecModel,
    TransformerModel,
)
from PipelineTS.pipeline import SmartRouter


NN_CLASSES = [
    DLinearModel,
    NLinearModel,
    NBeatsModel,
    NHitsModel,
    TiDEModel,
    TCNModel,
    PatchRNNModel,
    StackingRNNModel,
    Time2VecModel,
    GAUModel,
    TransformerModel,
    TFTModel,
    DeepARModel,
    ITransformerModel,
    SRSNetModel,
]


NN_MODELS = [
    "d_linear",
    "n_linear",
    "n_beats",
    "n_hits",
    "tcn",
    "tft",
    "gau",
    "stacking_rnn",
    "time2vec",
    "transformer",
    "tide",
    "patch_rnn",
    "itransformer",
    "srs_net",
    "deepar",
]


def check_wrapper_defaults():
    for cls in NN_CLASSES:
        sig = inspect.signature(cls)
        epochs = sig.parameters["epochs"].default
        patience = sig.parameters["patience"].default
        assert epochs == 1500, f"{cls.__name__} epochs={epochs}, expected 1500"
        assert patience == 80, f"{cls.__name__} patience={patience}, expected 80"
    print(f"[PASS] wrapper defaults: {len(NN_CLASSES)} models epochs=1500 patience=80")


def check_smartrouter_caps():
    base = {}
    for model in NN_MODELS:
        base[f"{model}__epochs"] = 3000
        base[f"{model}__patience"] = 150

    expected = {
        "fast": (25, 8),
        "medium_quality": (800, 80),
        "high_quality": (1500, 120),
    }
    for preset, (expected_epochs, expected_patience) in expected.items():
        router = SmartRouter(time_col="date", target_col="value", preset=preset, verbose=False)
        capped, _ = router._apply_training_budget_caps(base, models=NN_MODELS)
        for model in NN_MODELS:
            epochs = capped[f"{model}__epochs"]
            patience = capped[f"{model}__patience"]
            assert epochs == expected_epochs, f"{preset}:{model} epochs={epochs}, expected {expected_epochs}"
            assert patience == expected_patience, f"{preset}:{model} patience={patience}, expected {expected_patience}"
    router = SmartRouter(time_col="date", target_col="value", preset="medium_quality", verbose=False)
    capped, _ = router._apply_training_budget_caps(base, models=NN_MODELS, per_model_budget=20.0)
    assert capped["n_beats__epochs"] == 100
    assert capped["n_beats__patience"] == 15
    print("[PASS] SmartRouter caps: fast=25/8 medium=800/80 high=1500/120 tight_budget=100/15")


def main():
    check_wrapper_defaults()
    check_smartrouter_caps()


if __name__ == "__main__":
    main()
