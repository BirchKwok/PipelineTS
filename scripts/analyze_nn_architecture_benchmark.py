import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = [
    "tmp_benchmark_results/nn_arch_smoke.json",
    "tmp_benchmark_results/nn_arch_builtins.json",
    "tmp_benchmark_results/nn_arch_enhancements.json",
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
        if p.suffix == ".jsonl":
            data = [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
        for row in data:
            item = dict(row)
            item["source"] = p.name
            rows.append(item)
    return rows


def finite_mean(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if len(arr) else np.nan


def fmt(value):
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.4f}"


def summarize_errors(rows):
    bad = [r for r in rows if r.get("status") != "ok"]
    if not bad:
        print("\nErrors/skips: none")
        return
    counts = Counter((r.get("status"), r.get("model"), r.get("variant"), r.get("error")) for r in bad)
    print("\nErrors/skips")
    print("============")
    for (status, model, variant, error), count in counts.most_common(20):
        print(f"{count:>4} {status:<8} {model:<16} {variant:<24} {error}")


def summarize_ignored_kwargs(rows):
    counts = Counter()
    for row in rows:
        for key in row.get("ignored_kwargs") or []:
            if key not in {"backend"}:
                counts[(row.get("model"), row.get("variant"), key)] += 1
    if not counts:
        print("\nIgnored kwargs: none")
        return
    print("\nIgnored kwargs")
    print("==============")
    for (model, variant, key), count in counts.most_common(30):
        print(f"{count:>4} {model:<16} {variant:<24} {key}")


def baseline_key(row):
    return (row.get("source"), row.get("dataset"), row.get("model"), row.get("seed"))


def summarize_relative(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    baselines = {baseline_key(r): r for r in ok if r.get("variant") == "baseline"}
    rel = []
    for row in ok:
        if row.get("variant") == "baseline":
            continue
        base = baselines.get(baseline_key(row))
        if not base or not base.get("smape") or not base.get("mae"):
            continue
        item = dict(row)
        item["mae_ratio"] = row["mae"] / base["mae"]
        item["smape_ratio"] = row["smape"] / base["smape"]
        item["wmape_ratio"] = row["wmape"] / base["wmape"] if base.get("wmape") else np.nan
        item["fit_ratio"] = row["fit_seconds"] / base["fit_seconds"] if base.get("fit_seconds") else np.nan
        if row.get("trainable_params") and base.get("trainable_params"):
            item["param_ratio"] = row["trainable_params"] / base["trainable_params"]
        else:
            item["param_ratio"] = np.nan
        rel.append(item)
    if not rel:
        print("\nRelative summary: no baseline-matched rows")
        return

    variant_groups = defaultdict(list)
    model_variant_groups = defaultdict(list)
    winners = Counter()
    by_group = defaultdict(list)
    for row in ok:
        by_group[baseline_key(row)].append(row)
    for group_rows in by_group.values():
        best = min(group_rows, key=lambda r: r.get("smape", np.inf))
        winners[best.get("variant")] += 1
    for row in rel:
        variant_groups[row["variant"]].append(row)
        model_variant_groups[(row["model"], row["variant"])].append(row)

    print("\nBest variants by matched group")
    print("==============================")
    print(", ".join(f"{k}:{v}" for k, v in winners.most_common()))

    print("\nVariant aggregate vs baseline")
    print("=============================")
    print(f"{'variant':<26} {'n':>4} {'wins':>5} {'mean_smape':>12} {'median_smape':>13} {'mean_mae':>10} {'mean_fit':>10} {'mean_param':>11}")
    for variant, items in sorted(variant_groups.items()):
        wins = sum(1 for item in items if item["smape_ratio"] < 1.0)
        print(
            f"{variant:<26} {len(items):>4} {wins:>5} "
            f"{fmt(finite_mean([i['smape_ratio'] for i in items])):>12} "
            f"{fmt(float(np.median([i['smape_ratio'] for i in items]))):>13} "
            f"{fmt(finite_mean([i['mae_ratio'] for i in items])):>10} "
            f"{fmt(finite_mean([i['fit_ratio'] for i in items])):>10} "
            f"{fmt(finite_mean([i['param_ratio'] for i in items])):>11}"
        )

    print("\nPer-model best variant by mean SMAPE ratio")
    print("==========================================")
    model_to_items = defaultdict(list)
    for key, items in model_variant_groups.items():
        model_to_items[key[0]].append((key[1], items))
    for model in sorted(model_to_items):
        candidates = []
        for variant, items in model_to_items[model]:
            candidates.append((finite_mean([i["smape_ratio"] for i in items]), variant, len(items)))
        candidates = sorted(candidates, key=lambda x: x[0])[:5]
        text = ", ".join(f"{variant}:{fmt(score)}(n={n})" for score, variant, n in candidates)
        print(f"{model:<16} {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS))
    args = parser.parse_args()
    rows = load_rows(parse_csv(args.inputs))
    print(f"Loaded rows: {len(rows)}")
    if not rows:
        raise SystemExit(1)
    summarize_errors(rows)
    summarize_ignored_kwargs(rows)
    summarize_relative(rows)


if __name__ == "__main__":
    main()
