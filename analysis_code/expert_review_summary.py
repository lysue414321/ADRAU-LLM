"""Summarise blinded expert ratings using strict and flexible criteria."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paired_antibiotic_analysis import parse_age_years, read_table


def summarise(group: pd.DataFrame, score_column: str, dimension: str) -> dict[str, object]:
    score = pd.to_numeric(group[score_column], errors="coerce").dropna()
    return {
        "dimension": dimension,
        "evaluable_n": len(score),
        "strict_n": int(score.eq(2).sum()),
        "strict_pct": score.eq(2).mean() * 100 if len(score) else np.nan,
        "flexible_n": int(score.ge(1).sum()),
        "flexible_pct": score.ge(1).mean() * 100 if len(score) else np.nan,
        "inappropriate_n": int(score.eq(0).sum()),
        "inappropriate_pct": score.eq(0).mean() * 100 if len(score) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--decision-score-column", required=True)
    parser.add_argument("--choice-score-column")
    parser.add_argument("--category-column")
    parser.add_argument("--age-column")
    parser.add_argument("--source-column")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = read_table(args.input).copy()
    data["_source"] = data[args.source_column].astype(str) if args.source_column else "ADRAU-LLM"
    data["_category"] = (
        data[args.category_column].astype(str).str.strip().str.upper()
        if args.category_column
        else "All"
    )
    if args.age_column:
        age = parse_age_years(data[args.age_column])
        data["_age_group"] = np.where(age >= 18, "Adult", "Paediatric")
        data.loc[age.isna(), "_age_group"] = "Age missing"
    else:
        data["_age_group"] = "All ages"

    rows = []
    grouping_sets = [
        ("Overall", ["_source"]),
        ("BMJ category", ["_source", "_category"]),
        ("Age group", ["_source", "_age_group"]),
    ]
    for analysis, columns in grouping_sets:
        for keys, group in data.groupby(columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            labels = dict(zip(columns, keys))
            common = {
                "analysis": analysis,
                "source": labels.get("_source", "ADRAU-LLM"),
                "category": labels.get("_category", "All"),
                "age_group": labels.get("_age_group", "All ages"),
            }
            rows.append({**common, **summarise(group, args.decision_score_column, "Antibiotic-use decision")})
            if args.choice_score_column:
                rows.append({**common, **summarise(group, args.choice_score_column, "Antibiotic choice")})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
