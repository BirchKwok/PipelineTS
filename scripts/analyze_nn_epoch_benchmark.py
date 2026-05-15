import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    "tmp_benchmark_results/nn_default_epochs_subset.json",
    "tmp_benchmark_results/nn_default_epochs_remaining.json",
    "tmp_benchmark_results/nn_default_epochs_patience.json",
]


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_rows(paths):
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"SKIP missing {p}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for row in data:
            if row.get("status") == "ok":
                item = dict(row)
                item["source"] = p.name
                rows.append(item)
    return rows


def fmt(value):
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.4f}"


def summarize_rows(title, rows, base_profile):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"])].append(row)
    wins = Counter()
    ratios = defaultdict(list)
    fit_seconds = defaultdict(list)
    trained_epochs = defaultdict(list)
    per_model = defaultdict(lambda: defaultdict(list))
    for key, items in groups.items():
        best = min(items, key=lambda x: x["smape"])
        wins[best["profile"]] += 1
        base = next((x for x in items if x["profile"] == base_profile), None)
        if base is None or base["smape"] <= 0:
            continue
        for item in items:
            profile = item["profile"]
            ratio = item["smape"] / base["smape"]
            ratios[profile].append(ratio)
            fit_seconds[profile].append(item["fit_seconds"])
            trained_epochs[profile].append(item["trained_epochs"])
            per_model[item["model"]][profile].append(ratio)
    print(f"\n{title}")
    print("=" * len(title))
    print(f"groups={len(groups)} rows={len(rows)} base={base_profile}")
    print("wins=" + ", ".join(f"{k}:{v}" for k, v in wins.most_common()))
    print(f"{'profile':<12} {'n':>4} {'mean_ratio':>11} {'median_ratio':>13} {'mean_fit_s':>11} {'median_ep':>10}")
    for profile in sorted(ratios):
        vals = np.asarray(ratios[profile], dtype=float)
        fits = np.asarray(fit_seconds[profile], dtype=float)
        eps = np.asarray(trained_epochs[profile], dtype=float)
        print(f"{profile:<12} {len(vals):>4} {np.mean(vals):>11.4f} {np.median(vals):>13.4f} {np.mean(fits):>11.2f} {np.median(eps):>10.1f}")
    if per_model:
        profiles = sorted({p for model_data in per_model.values() for p in model_data})
        print("\nPer-model mean relative SMAPE")
        print(f"{'model':<16} " + " ".join(f"{p:>11}" for p in profiles) + "   best")
        for model in sorted(per_model):
            values = {}
            for profile in profiles:
                vals = per_model[model].get(profile)
                values[profile] = float(np.mean(vals)) if vals else np.nan
            best_profile = min((p for p in profiles if np.isfinite(values[p])), key=lambda p: values[p])
            print(f"{model:<16} " + " ".join(f"{fmt(values[p]):>11}" for p in profiles) + f"   {best_profile}")
    return ratios, per_model, wins


def recommend(rows):
    broad_sources = {"nn_default_epochs_subset.json", "nn_default_epochs_remaining.json"}
    patience_sources = {"nn_default_epochs_patience.json"}
    broad = [r for r in rows if r["source"] in broad_sources]
    patience = [r for r in rows if r["source"] in patience_sources]
    print("\nRecommendations")
    print("===============")
    if broad:
        ratios, per_model, wins = summarize_rows("Broad epoch sweep", broad, "cap300")
        candidate_profiles = [p for p in ratios if p != "cap300"]
        if candidate_profiles:
            best_profile = min(candidate_profiles, key=lambda p: float(np.mean(ratios[p])))
            print(f"\nBest broad profile by mean relative SMAPE: {best_profile}")
    if patience:
        ratios, per_model, wins = summarize_rows("Patience sweep", patience, "cap800")
        candidate_profiles = [p for p in ratios if p != "cap800"]
        if candidate_profiles:
            best_profile = min(candidate_profiles, key=lambda p: float(np.mean(ratios[p])))
            print(f"\nBest patience profile by mean relative SMAPE vs cap800: {best_profile}")
    print("\nSuggested code policy")
    print("- NN wrapper defaults: use a convergence-friendly budget around epochs=1500 and patience=80 for most NN wrappers.")
    print("- SmartRouter medium_quality: do not cap NN epochs at 300; use at least 800 epochs and patience around 80.")
    print("- SmartRouter high_quality: allow long runs around 1500 epochs and patience around 120.")
    print("- SmartRouter fast: keep a low cap for quick screening only.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS))
    args = parser.parse_args()
    rows = load_rows(parse_csv(args.inputs))
    print(f"Loaded ok rows: {len(rows)}")
    if not rows:
        raise SystemExit(1)
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    for source, items in sorted(by_source.items()):
        profiles = {row["profile"] for row in items}
        base = "cap300" if "cap300" in profiles else "cap800"
        summarize_rows(source, items, base)
    recommend(rows)


if __name__ == "__main__":
    main()
