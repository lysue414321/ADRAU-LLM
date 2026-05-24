#!/usr/bin/env python3
"""
ADRAU-LLM EHR Data Preprocessing Pipeline
=========================================
Preprocesses outpatient electronic health record (EHR) data for downstream
antibiotic prescription analysis and LLM fine-tuning.

Functions:
    clean_ehr_records()       -- Remove duplicates, filter missing diagnosis fields
    enrich_lab_tests()        -- Add reference ranges to lab test results
    merge_icd_subcodes()      -- Collapse ICD-10 subcodes to parent codes
    tiered_resampling()       -- Balance dataset via tiered up/down-sampling

Usage:
    python data_preprocessing.py --input data/raw_ehr.csv --output data/processed_ehr.csv
    python data_preprocessing.py --input data/raw_ehr.csv --output data/processed_ehr.csv \
        --lab_ref data/lab_reference_ranges.json --icd_mapping data/icd_mapping.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ehr_preprocessing")


# ===================================================================
# Core Functions
# ===================================================================

def clean_ehr_records(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    drop_duplicates_subset: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Clean raw EHR records by removing duplicates and rows with missing
    diagnosis fields.

    Parameters
    ----------
    df : pd.DataFrame
        Raw EHR data.
    required_columns : list[str], optional
        Columns that must be non-null for a record to be retained.
        Defaults to ["diagnosis_code", "diagnosis_name", "visit_date"].
    drop_duplicates_subset : list[str], optional
        Columns used to identify duplicate records.
        Defaults to ["patient_id", "visit_date", "diagnosis_code"].

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with duplicates removed and required fields present.
    """
    if required_columns is None:
        required_columns = ["diagnosis_code", "diagnosis_name", "visit_date"]

    if drop_duplicates_subset is None:
        drop_duplicates_subset = ["patient_id", "visit_date", "diagnosis_code"]

    n_before = len(df)
    logger.info("Starting EHR record cleaning. Initial records: %d", n_before)

    # --- Step 1: Drop fully duplicated rows ---
    df = df.drop_duplicates(subset=drop_duplicates_subset, keep="first")
    n_after_dedup = len(df)
    logger.info(
        "Deduplication: %d records dropped, %d remaining.",
        n_before - n_after_dedup,
        n_after_dedup,
    )

    # --- Step 2: Filter rows with missing required fields ---
    # Ensure all required columns exist in the dataframe
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Required columns missing from input data: {missing_cols}"
        )

    mask = df[required_columns].notnull().all(axis=1)
    df = df.loc[mask].copy()
    n_after_filter = len(df)
    logger.info(
        "Missing-value filter: %d records dropped (%d records kept).",
        n_after_dedup - n_after_filter,
        n_after_filter,
    )

    # --- Step 3: Strip whitespace and normalize diagnosis_code ---
    if "diagnosis_code" in df.columns:
        df["diagnosis_code"] = (
            df["diagnosis_code"].astype(str).str.strip().str.upper()
        )

    logger.info("EHR record cleaning complete. Final records: %d", len(df))
    return df


def enrich_lab_tests(
    df: pd.DataFrame,
    lab_ref_path: Optional[str] = None,
) -> pd.DataFrame:
    """Enrich lab test records by joining reference range metadata.

    Adds columns ``lower_limit``, ``upper_limit``, and ``unit`` to each
    lab test row based on a reference range lookup table. If no reference
    file is provided, a set of common default ranges is applied for common
    tests (WBC, CRP, PCT, SCr, ALT, AST).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame expected to contain ``test_name`` (str) and
        ``test_result_value`` (float) columns.
    lab_ref_path : str, optional
        Path to a JSON file mapping test_name to {lower, upper, unit}.
        Expected format: {"WBC": {"lower": 3.5, "upper": 9.5, "unit": "10^9/L"}, ...}

    Returns
    -------
    pd.DataFrame
        The input DataFrame with ``lower_limit``, ``upper_limit``, and ``unit``
        columns added.
    """
    logger.info("Enriching lab tests with reference ranges.")

    if lab_ref_path and os.path.exists(lab_ref_path):
        with open(lab_ref_path, "r", encoding="utf-8") as fh:
            ref_table = json.load(fh)
        ref_df = pd.DataFrame(ref_table).T
        ref_df.index.name = "test_name"
        ref_df = ref_df.reset_index()
        logger.info("Loaded reference ranges from %s", lab_ref_path)
    else:
        # Fallback: built-in defaults for common labs
        logger.info("No lab reference file provided; using built-in defaults.")
        defaults = {
            "WBC": {"lower": 3.5, "upper": 9.5, "unit": "10^9/L"},
            "NEUT": {"lower": 1.8, "upper": 6.3, "unit": "10^9/L"},
            "CRP": {"lower": 0.0, "upper": 5.0, "unit": "mg/L"},
            "PCT": {"lower": 0.0, "upper": 0.05, "unit": "ng/mL"},
            "SCr": {"lower": 44.0, "upper": 133.0, "unit": "umol/L"},
            "ALT": {"lower": 7.0, "upper": 40.0, "unit": "U/L"},
            "AST": {"lower": 13.0, "upper": 35.0, "unit": "U/L"},
            "TBIL": {"lower": 3.4, "upper": 17.1, "unit": "umol/L"},
            "BUN": {"lower": 2.9, "upper": 8.2, "unit": "mmol/L"},
        }
        ref_df = pd.DataFrame(defaults).T.reset_index()
        ref_df.columns = ["test_name", "lower", "upper", "unit"]

    # Join reference ranges
    df = df.merge(
        ref_df,
        on="test_name",
        how="left",
        suffixes=("", "_ref"),
    )

    df["lower_limit"] = df.get("lower", np.nan)
    df["upper_limit"] = df.get("upper", np.nan)
    df["unit"] = df.get("unit", None)

    # Drop redundant columns if they exist
    for col in ["lower", "upper", "lower_ref", "upper_ref"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Flag abnormal results
    df["is_abnormal"] = (
        (df["test_result_value"] < df["lower_limit"])
        | (df["test_result_value"] > df["upper_limit"])
    )

    n_abnormal = df["is_abnormal"].sum()
    logger.info(
        "Lab test enrichment complete. %d / %d results flagged abnormal.",
        n_abnormal,
        len(df),
    )
    return df


def merge_icd_subcodes(
    df: pd.DataFrame,
    code_column: str = "diagnosis_code",
    mapping_path: Optional[str] = None,
) -> pd.DataFrame:
    """Collapse ICD-10 subcodes to their parent codes.

    For example, J18.0 through J18.9 all map to J18 (Pneumonia, organism
    unspecified). This reduces granularity to a clinically relevant level
    for antibiotic prescribing analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a diagnosis code column.
    code_column : str
        Name of the column containing ICD-10 codes.
    mapping_path : str, optional
        Path to a JSON file mapping subcodes to parent codes. If omitted,
        a built-in rule-based mapping is applied (truncate at the decimal
        point for codes in the J00-J99 respiratory chapter).

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``icd_parent_code`` column.
    """
    logger.info("Collapsing ICD-10 subcodes to parent codes.")

    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as fh:
            icd_map = json.load(fh)
        df["icd_parent_code"] = (
            df[code_column].astype(str).str.strip().str.upper().map(icd_map)
        )
        # Fill unmapped codes with themselves
        df["icd_parent_code"] = df["icd_parent_code"].fillna(
            df[code_column].astype(str).str.strip().str.upper()
        )
        logger.info("Loaded ICD mapping from %s", mapping_path)
    else:
        # Built-in rule: for respiratory chapter (J00-J99), strip subcode
        logger.info("No ICD mapping provided; using rule-based collapse.")
        icd = df[code_column].astype(str).str.strip().str.upper()

        def _collapse(code: str) -> str:
            """Strip subcode after '.' for J-codes; keep other codes as-is."""
            if code and code[0] == "J":
                return code.split(".")[0] if "." in code else code
            return code

        df["icd_parent_code"] = icd.apply(_collapse)

    n_changed = (df["icd_parent_code"] != df[code_column]).sum()
    logger.info(
        "ICD subcode collapse complete. %d / %d codes modified.",
        n_changed,
        len(df),
    )
    return df


def tiered_resampling(
    df: pd.DataFrame,
    label_column: str = "icd_parent_code",
    upper_threshold: int = 10000,
    lower_threshold: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Balance class distribution via tiered resampling.

    Strategy:
        - Classes with > ``upper_threshold`` samples are randomly downsampled
          to ``upper_threshold``.
        - Classes with < ``lower_threshold`` samples are oversampled (with
          replacement) to ``lower_threshold``.
        - Classes with counts between the two thresholds are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a label column.
    label_column : str
        Column containing class labels for stratification.
    upper_threshold : int
        Maximum number of samples per class after downsampling.
    lower_threshold : int
        Minimum number of samples per class after upsampling.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame.
    """
    logger.info(
        "Starting tiered resampling (upper=%d, lower=%d, keep=%d-%d).",
        upper_threshold,
        lower_threshold,
        lower_threshold,
        upper_threshold,
    )

    rng = np.random.default_rng(random_state)
    class_counts = df[label_column].value_counts()
    logger.info("Class distribution before resampling:\n%s", class_counts)

    resampled_parts: list[pd.DataFrame] = []

    for label, count in class_counts.items():
        subset = df[df[label_column] == label]

        if count > upper_threshold:
            # Downsample
            indices = rng.choice(
                subset.index,
                size=upper_threshold,
                replace=False,
            )
            resampled_parts.append(df.loc[indices])
            logger.debug(
                "%s: downsampled %d -> %d", label, count, upper_threshold
            )

        elif count < lower_threshold:
            # Upsample with replacement
            indices = rng.choice(
                subset.index,
                size=lower_threshold,
                replace=True,
            )
            resampled_parts.append(df.loc[indices])
            logger.debug(
                "%s: upsampled %d -> %d", label, count, lower_threshold
            )

        else:
            # Keep as-is
            resampled_parts.append(subset)
            logger.debug("%s: kept at %d (within range)", label, count)

    result = pd.concat(resampled_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    logger.info(
        "Tiered resampling complete. Records: %d -> %d",
        len(df),
        len(result),
    )
    return result


# ===================================================================
# Orchestration
# ===================================================================

def main() -> None:
    """Orchestrate the EHR preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="ADRAU-LLM: Preprocess outpatient EHR data."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV file (raw EHR records).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output CSV file (processed records).",
    )
    parser.add_argument(
        "--lab_ref",
        default=None,
        help="Path to JSON file containing lab reference ranges.",
    )
    parser.add_argument(
        "--icd_mapping",
        default=None,
        help="Path to JSON file mapping ICD subcodes to parent codes.",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Skip tiered resampling step.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=int,
        default=10000,
        help="Upper count threshold for downsampling (default: 10000).",
    )
    parser.add_argument(
        "--lower-threshold",
        type=int,
        default=3000,
        help="Lower count threshold for upsampling (default: 3000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    logger.info("Loading raw EHR data from %s", args.input)
    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Loaded %d records, %d columns.", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # Step 1: Clean records
    # ------------------------------------------------------------------
    df = clean_ehr_records(df)

    # ------------------------------------------------------------------
    # Step 2: Merge ICD subcodes
    # ------------------------------------------------------------------
    df = merge_icd_subcodes(df, mapping_path=args.icd_mapping)

    # ------------------------------------------------------------------
    # Step 3: Enrich lab tests (if lab data present)
    # ------------------------------------------------------------------
    if "test_name" in df.columns:
        df = enrich_lab_tests(df, lab_ref_path=args.lab_ref)
    else:
        logger.info(
            "No 'test_name' column found; skipping lab test enrichment."
        )

    # ------------------------------------------------------------------
    # Step 4: Tiered resampling
    # ------------------------------------------------------------------
    if not args.no_resample:
        df = tiered_resampling(
            df,
            label_column="icd_parent_code",
            upper_threshold=args.upper_threshold,
            lower_threshold=args.lower_threshold,
            random_state=args.seed,
        )
    else:
        logger.info("Resampling skipped (--no-resample flag).")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(
        "Preprocessed EHR data saved to %s (%d records).",
        args.output,
        len(df),
    )


if __name__ == "__main__":
    main()
