"""Paired antibiotic-use analyses used in the ADRAU-LLM revision.

The script expects one row per case, so model and physician decisions are
paired by construction. It reports antibiotic-use rates, paired rate
differences, McNemar tests, percentile-bootstrap confidence intervals, and
BMJ Never/Always inappropriate-decision summaries, including age strata.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_binary(series: pd.Series) -> pd.Series:
    yes = {"1", "1.0", "true", "yes", "y", "use", "used", "\u662f"}
    no = {"0", "0.0", "false", "no", "n", "non-use", "not used", "\u5426"}

    def convert(value: object) -> float:
        if pd.isna(value):
            return np.nan
        token = str(value).strip().lower()
        if token in yes:
            return 1.0
        if token in no:
            return 0.0
        raise ValueError(f"Unrecognised binary value: {value!r}")

    return series.map(convert)


def parse_age_years(series: pd.Series) -> pd.Series:
    def convert(value: object) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)
        token = str(value).strip().lower()
        match = re.search(r"\d+(?:\.\d+)?", token)
        if not match:
            return np.nan
        number = float(match.group())
        if "month" in token or "\u6708" in token:
            return number / 12.0
        if "day" in token or "\u5929" in token:
            return number / 365.25
        return number

    return series.map(convert)


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    discordant_01 = int(np.sum((a == 0) & (b == 1)))
    discordant_10 = int(np.sum((a == 1) & (b == 0)))
    discordant = discordant_01 + discordant_10
    if discordant == 0:
        return 1.0, discordant_01, discordant_10
    if discordant < 25:
        low = min(discordant_01, discordant_10)
        tail = sum(math.comb(discordant, k) for k in range(low + 1)) / (2**discordant)
        return min(1.0, 2 * tail), discordant_01, discordant_10
    statistic = (abs(discordant_01 - discordant_10) - 1) ** 2 / discordant
    return math.erfc(math.sqrt(statistic / 2)), discordant_01, discordant_10


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(a), len(a))
        a_mean = float(a[indices].mean())
        b_mean = float(b[indices].mean())
        if metric == "difference":
            estimates.append((a_mean - b_mean) * 100)
        elif b_mean > 0:
            estimates.append((b_mean - a_mean) / b_mean * 100)
    if not estimates:
        return np.nan, np.nan
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def inappropriate(category: pd.Series, use: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=category.index, dtype=float)
    result.loc[category.eq("N") & use.notna()] = use.loc[category.eq("N") & use.notna()]
    result.loc[category.eq("A") & use.notna()] = 1 - use.loc[category.eq("A") & use.notna()]
    return result


def analyse_rates(
    data: pd.DataFrame,
    comparator: str,
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in ["Overall", "N", "S", "A"]:
        subset = data if group == "Overall" else data[data["category"].eq(group)]
        subset = subset.dropna(subset=["adrau", comparator])
        if subset.empty:
            continue
        a = subset["adrau"].to_numpy(dtype=int)
        b = subset[comparator].to_numpy(dtype=int)
        low, high = bootstrap_ci(a, b, "difference", iterations, rng)
        p_value, d01, d10 = mcnemar_p(a, b)
        interpretation = {
            "Overall": "Antibiotic-use or recommendation frequency",
            "N": "Lower rates indicate fewer potential overuse decisions",
            "S": "Descriptive frequency; diagnosis alone does not determine appropriateness",
            "A": "Higher rates indicate fewer potential underuse decisions",
        }[group]
        rows.append(
            {
                "comparator": comparator,
                "group": group,
                "n": len(subset),
                "adrau_rate_pct": a.mean() * 100,
                "comparator_rate_pct": b.mean() * 100,
                "difference_pp": (a.mean() - b.mean()) * 100,
                "difference_95ci_low_pp": low,
                "difference_95ci_high_pp": high,
                "mcnemar_p": p_value,
                "discordant_0_1": d01,
                "discordant_1_0": d10,
                "interpretation": interpretation,
            }
        )
    return rows


def analyse_errors(
    data: pd.DataFrame,
    comparator: str,
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    strata = {"All ages": data}
    if "age_group" in data:
        strata.update({name: group for name, group in data.groupby("age_group", dropna=True)})
    for age_group, subset in strata.items():
        subset = subset[subset["category"].isin(["N", "A"])].copy()
        a_error = inappropriate(subset["category"], subset["adrau"])
        b_error = inappropriate(subset["category"], subset[comparator])
        valid = a_error.notna() & b_error.notna()
        a = a_error[valid].to_numpy(dtype=int)
        b = b_error[valid].to_numpy(dtype=int)
        if len(a) == 0:
            continue
        low, high = bootstrap_ci(a, b, "relative_reduction", iterations, rng)
        p_value, d01, d10 = mcnemar_p(a, b)
        comparator_errors = int(b.sum())
        reduction = (
            (comparator_errors - int(a.sum())) / comparator_errors * 100
            if comparator_errors
            else np.nan
        )
        rows.append(
            {
                "comparator": comparator,
                "age_group": age_group,
                "n_never_always": len(a),
                "adrau_inappropriate_n": int(a.sum()),
                "comparator_inappropriate_n": comparator_errors,
                "adrau_inappropriate_pct": a.mean() * 100,
                "comparator_inappropriate_pct": b.mean() * 100,
                "relative_error_reduction_pct": reduction,
                "reduction_95ci_low_pct": low,
                "reduction_95ci_high_pct": high,
                "mcnemar_p": p_value,
                "discordant_0_1": d01,
                "discordant_1_0": d10,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--category-column", required=True)
    parser.add_argument("--adrau-column", required=True)
    parser.add_argument("--base-column")
    parser.add_argument("--physician-column")
    parser.add_argument("--age-column")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = read_table(args.input)
    data = pd.DataFrame(
        {
            "category": raw[args.category_column].astype(str).str.strip().str.upper(),
            "adrau": parse_binary(raw[args.adrau_column]),
        }
    )
    comparators: list[str] = []
    for name, column in [("base", args.base_column), ("physician", args.physician_column)]:
        if column:
            data[name] = parse_binary(raw[column])
            comparators.append(name)
    if not comparators:
        raise ValueError("Provide --base-column and/or --physician-column")
    if args.age_column:
        age = parse_age_years(raw[args.age_column])
        data["age_group"] = np.where(age >= 18, "Adult", "Paediatric")
        data.loc[age.isna(), "age_group"] = np.nan

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rate_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for comparator in comparators:
        rate_rows.extend(analyse_rates(data, comparator, args.iterations, rng))
        error_rows.extend(analyse_errors(data, comparator, args.iterations, rng))
    pd.DataFrame(rate_rows).to_csv(args.output_dir / "antibiotic_rate_comparisons.csv", index=False)
    pd.DataFrame(error_rows).to_csv(args.output_dir / "bmj_never_always_error_reduction.csv", index=False)


if __name__ == "__main__":
    main()
