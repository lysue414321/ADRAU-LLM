"""Bootstrap and age-stratified analyses for the 40-category diagnosis task."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paired_antibiotic_analysis import mcnemar_p, parse_age_years, read_table


def effective_prediction(true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    included = (predictions == true[:, None]).any(axis=1)
    return np.where(included, true, predictions[:, 0])


def metrics(true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    effective = effective_prediction(true, predictions)
    f1, support, precision, recall = per_class_metrics(true, effective)
    weights = support / support.sum()
    return {
        "accuracy_or_inclusion": float(np.mean(true == effective)),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
    }


def per_class_metrics(
    true: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.unique(true)
    support = np.array([np.sum(true == label) for label in labels], dtype=float)
    predicted = np.array([np.sum(prediction == label) for label in labels], dtype=float)
    correct = np.array(
        [np.sum((true == label) & (prediction == label)) for label in labels],
        dtype=float,
    )
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(correct),
        where=(precision + recall) > 0,
    )
    return f1, support, precision, recall


def bootstrap_metric(
    true: np.ndarray,
    adrau: np.ndarray,
    base: np.ndarray,
    key: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    adrau_values = []
    base_values = []
    differences = []
    for _ in range(iterations):
        indices = rng.integers(0, len(true), len(true))
        a_value = metrics(true[indices], adrau[indices])[key]
        b_value = metrics(true[indices], base[indices])[key]
        adrau_values.append(a_value)
        base_values.append(b_value)
        differences.append(a_value - b_value)
    a_low, a_high = np.percentile(adrau_values, [2.5, 97.5])
    d_low, d_high = np.percentile(differences, [2.5, 97.5])
    return float(a_low), float(a_high), float(d_low), float(d_high)


def analyse_stratum(
    true: np.ndarray,
    adrau: np.ndarray,
    base: np.ndarray,
    label: str,
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows = []
    for level, width in [("Top-1", 1), ("Top-3", min(3, adrau.shape[1], base.shape[1]))]:
        a_pred = adrau[:, :width]
        b_pred = base[:, :width]
        a_metrics = metrics(true, a_pred)
        b_metrics = metrics(true, b_pred)
        a_correct = (a_pred == true[:, None]).any(axis=1).astype(int)
        b_correct = (b_pred == true[:, None]).any(axis=1).astype(int)
        p_value, d01, d10 = mcnemar_p(a_correct, b_correct)
        for key in ["accuracy_or_inclusion", "weighted_precision", "weighted_recall", "weighted_f1"]:
            a_low, a_high, d_low, d_high = bootstrap_metric(
                true, a_pred, b_pred, key, iterations, rng
            )
            rows.append(
                {
                    "stratum": label,
                    "level": level,
                    "metric": key,
                    "n": len(true),
                    "adrau": a_metrics[key],
                    "adrau_95ci_low": a_low,
                    "adrau_95ci_high": a_high,
                    "base": b_metrics[key],
                    "difference": a_metrics[key] - b_metrics[key],
                    "difference_95ci_low": d_low,
                    "difference_95ci_high": d_high,
                    "mcnemar_p_correctness": p_value,
                    "discordant_0_1": d01,
                    "discordant_1_0": d10,
                }
            )
    return rows


def per_class_f1(
    true: np.ndarray, predictions: np.ndarray
) -> tuple[pd.Series, pd.Series]:
    effective = effective_prediction(true, predictions)
    labels = np.unique(true)
    f1, support, _, _ = per_class_metrics(true, effective)
    return pd.Series(f1, index=labels), pd.Series(support, index=labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--true-column", required=True)
    parser.add_argument("--adrau-columns", nargs="+", required=True)
    parser.add_argument("--base-columns", nargs="+", required=True)
    parser.add_argument("--age-column")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = read_table(args.input)
    required = [args.true_column, *args.adrau_columns, *args.base_columns]
    valid = raw[required].notna().all(axis=1)
    raw = raw.loc[valid].reset_index(drop=True)
    true = raw[args.true_column].astype(str).to_numpy()
    adrau = raw[args.adrau_columns].astype(str).to_numpy()
    base = raw[args.base_columns].astype(str).to_numpy()
    if adrau.shape[1] != base.shape[1]:
        raise ValueError("ADRAU and base prediction column counts must match")

    strata = {"Overall": np.ones(len(raw), dtype=bool)}
    if args.age_column:
        age = parse_age_years(raw[args.age_column])
        strata["Adult"] = (age >= 18).to_numpy()
        strata["Paediatric"] = (age < 18).to_numpy()

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for label, mask in strata.items():
        rows.extend(analyse_stratum(true[mask], adrau[mask], base[mask], label, args.iterations, rng))

    a_top1, support = per_class_f1(true, adrau[:, :1])
    b_top1, _ = per_class_f1(true, base[:, :1])
    width = min(3, adrau.shape[1])
    a_top3, _ = per_class_f1(true, adrau[:, :width])
    b_top3, _ = per_class_f1(true, base[:, :width])
    per_class = pd.DataFrame(
        {
            "support": support,
            "adrau_top1_f1": a_top1,
            "base_top1_f1": b_top1,
            "delta_top1_f1": a_top1 - b_top1,
            "adrau_top3_f1": a_top3,
            "base_top3_f1": b_top3,
            "delta_top3_f1": a_top3 - b_top3,
        }
    )
    summary = pd.DataFrame(
        {
            "level": ["Top-1", "Top-3"],
            "mean_delta_f1": [per_class.delta_top1_f1.mean(), per_class.delta_top3_f1.mean()],
            "sd_delta_f1": [per_class.delta_top1_f1.std(ddof=1), per_class.delta_top3_f1.std(ddof=1)],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "diagnostic_bootstrap_metrics.csv", index=False)
    per_class.to_csv(args.output_dir / "diagnostic_per_class_f1.csv", index_label="diagnosis")
    summary.to_csv(args.output_dir / "diagnostic_delta_f1_mean_sd.csv", index=False)


if __name__ == "__main__":
    main()
