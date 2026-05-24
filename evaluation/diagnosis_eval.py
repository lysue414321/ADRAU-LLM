#!/usr/bin/env python3
"""
ADRAU-LLM Diagnosis Evaluation
===============================

Loads model predictions and physician ground-truth diagnoses from JSONL
files and computes a comprehensive suite of evaluation metrics:

    - Top-1 Accuracy
    - Top-3 Accuracy
    - Precision / Recall / Weighted F1 (per ICD-10 category and overall)
    - Confusion matrix (saved as CSV)
    - Error reduction relative to baseline model

Input Formats
-------------
Predictions JSONL (one JSON object per line):
    {
        "patient_id": "P001",
        "diagnoses": [
            {"rank": 1, "icd10_code": "J15.9", "diagnosis_name": "..."},
            {"rank": 2, "icd10_code": "J18.9", "diagnosis_name": "..."},
            {"rank": 3, "icd10_code": "J44.1", "diagnosis_name": "..."}
        ]
    }

Ground Truth JSONL:
    {
        "patient_id": "P001",
        "icd10_primary": "J15.9",
        "icd10_all": ["J15.9", "J44.1"]
    }

Usage:
    python diagnosis_eval.py \
        --predictions ./outputs/predictions.jsonl \
        --ground_truth ./data/diagnosis_gold.jsonl \
        --baseline_results ./baseline/diagnosis_baseline.jsonl \
        --output_dir ./evaluation/diagnosis_results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("adrau-llm.diagnosis_eval")


# ===================================================================
# ICD-10 Category Mapping
# ===================================================================

ICD10_CATEGORY_MAP = {
    # Infectious and parasitic diseases
    "A00-A09": ["A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"],
    "A15-A19": ["A15", "A16", "A17", "A18", "A19"],
    "A30-A49": ["A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39",
                  "A40", "A41", "A42", "A43", "A44", "A46", "A48", "A49"],
    # Respiratory
    "J00-J06": ["J00", "J01", "J02", "J03", "J04", "J05", "J06"],
    "J09-J18": ["J09", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18"],
    "J20-J22": ["J20", "J21", "J22"],
    "J40-J47": ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"],
    # Circulatory
    "I10-I15": ["I10", "I11", "I12", "I13", "I15"],
    "I20-I25": ["I20", "I21", "I22", "I23", "I24", "I25"],
    # Digestive
    "K20-K31": ["K20", "K21", "K22", "K25", "K26", "K27", "K28", "K29", "K30", "K31"],
    "K35-K38": ["K35", "K36", "K37", "K38"],
    # Genitourinary
    "N00-N08": ["N00", "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08"],
    "N10-N16": ["N10", "N11", "N12", "N13", "N15"],
    "N17-N19": ["N17", "N18", "N19"],
    "N30-N39": ["N30", "N31", "N32", "N34", "N35", "N36", "N39"],
    # Endocrine / Metabolic
    "E10-E14": ["E10", "E11", "E13", "E14"],
    # Symptoms / Signs
    "R00-R09": ["R00", "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R09"],
    "R50-R69": ["R50", "R51", "R52", "R53", "R54", "R55", "R56", "R57", "R58",
                  "R59", "R60", "R61", "R62", "R63", "R64", "R68"],
}


def get_icd10_category(icd10_code: str) -> str:
    """Map an ICD-10 code to its top-level category (e.g., J15.9 -> J09-J18).

    Strips the subcategory suffix before matching.
    """
    code = icd10_code.strip().upper()
    # Extract the base code (e.g., "J15.9" -> "J15")
    base = code.split(".")[0] if "." in code else code
    # Try exact prefix match
    for category, codes in ICD10_CATEGORY_MAP.items():
        if base in codes:
            return category
    # Fallback: match by first character + first digit
    if len(base) >= 2:
        prefix = base[:3] if len(base) >= 3 else base
        for category in ICD10_CATEGORY_MAP:
            cat_start = category.split("-")[0][:3]
            if prefix == cat_start:
                return category
    return "OTHER"


# ===================================================================
# Data loading
# ===================================================================

def load_predictions(jsonl_path: str) -> Dict[str, List[str]]:
    """Load predictions from JSONL.

    Returns
    -------
    dict[str, list[str]]
        Mapping from patient_id to ordered list of predicted ICD-10 codes
        (rank 1, 2, 3).
    """
    predictions: Dict[str, List[str]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                continue

            pid = rec.get("patient_id")
            if not pid:
                logger.warning("Missing patient_id at line %d; skipping", line_num)
                continue

            diags = rec.get("diagnoses", [])
            codes = [d.get("icd10_code", "").strip().upper() for d in diags]
            predictions[pid] = codes

    logger.info("Loaded %d predictions from %s", len(predictions), jsonl_path)
    return predictions


def load_ground_truth(jsonl_path: str) -> Dict[str, Set[str]]:
    """Load ground-truth diagnoses from JSONL.

    Returns
    -------
    dict[str, set[str]]
        Mapping from patient_id to set of all ground-truth ICD-10 codes.
    """
    ground_truth: Dict[str, Set[str]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                continue

            pid = rec.get("patient_id")
            if not pid:
                continue

            # Support both single primary and multi-label formats
            if "icd10_all" in rec:
                codes = {c.strip().upper() for c in rec["icd10_all"]}
            elif "icd10_primary" in rec:
                codes = {rec["icd10_primary"].strip().upper()}
            else:
                codes = set()
            ground_truth[pid] = codes

    logger.info("Loaded %d ground-truth records from %s", len(ground_truth), jsonl_path)
    return ground_truth


# ===================================================================
# Metrics computation
# ===================================================================

def compute_topk_accuracy(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k: int = 1,
) -> float:
    """Compute Top-K accuracy: fraction of patients where any ground-truth
    code appears in the top-K predictions."""
    correct = 0
    total = 0

    for pid, gt_codes in ground_truth.items():
        if pid not in predictions:
            continue
        pred_codes = predictions[pid][:k]
        if gt_codes & set(pred_codes):
            correct += 1
        total += 1

    if total == 0:
        return 0.0
    return correct / total


def compute_per_category_metrics(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    top_k: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 per ICD-10 category at Top-K level."""

    # Build per-category ground-truth sets
    cat_gt: Dict[str, Set[str]] = defaultdict(set)
    cat_pred: Dict[str, Set[str]] = defaultdict(set)

    for pid, gt_codes in ground_truth.items():
        if pid not in predictions:
            continue
        for code in gt_codes:
            cat = get_icd10_category(code)
            cat_gt[cat].add(pid)

        pred_codes = predictions[pid][:top_k]
        for code in pred_codes:
            cat = get_icd10_category(code)
            cat_pred[cat].add(pid)

    all_categories = sorted(set(cat_gt.keys()) | set(cat_pred.keys()))

    results: Dict[str, Dict[str, float]] = {}
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for cat in all_categories:
        # For each patient, check if this category is present in ground truth
        # and in predictions
        gts = cat_gt.get(cat, set())
        preds = cat_pred.get(cat, set())

        # Build binary labels across all patients
        all_pids = sorted(gts | preds)
        y_true_cat = [1 if pid in gts else 0 for pid in all_pids]
        y_pred_cat = [1 if pid in preds else 0 for pid in all_pids]

        if sum(y_true_cat) == 0:
            results[cat] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            continue

        precision = precision_score(y_true_cat, y_pred_cat, zero_division=0)
        recall = recall_score(y_true_cat, y_pred_cat, zero_division=0)
        f1 = f1_score(y_true_cat, y_pred_cat, zero_division=0)

        results[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(y_true_cat),
        }

        y_true_all.extend(y_true_cat)
        y_pred_all.extend(y_pred_cat)

    # Overall metrics
    if y_true_all:
        results["OVERALL"] = {
            "precision": round(precision_score(y_true_all, y_pred_all, zero_division=0), 4),
            "recall": round(recall_score(y_true_all, y_pred_all, zero_division=0), 4),
            "f1": round(f1_score(y_true_all, y_pred_all, zero_division=0), 4),
            "support": sum(y_true_all),
        }

    return results


def compute_weighted_f1(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    top_k: int = 1,
) -> float:
    """Compute macro-weighted F1 across all ICD-10 categories."""
    per_cat = compute_per_category_metrics(predictions, ground_truth, top_k)

    f1_values = []
    weights = []
    for cat, metrics in per_cat.items():
        if cat == "OVERALL":
            continue
        if metrics["support"] > 0:
            f1_values.append(metrics["f1"])
            weights.append(metrics["support"])

    if not weights:
        return 0.0

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return float(np.average(f1_values, weights=weights))


def build_confusion_matrix(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    top_k: int = 1,
) -> Tuple[np.ndarray, List[str]]:
    """Build a confusion matrix at the ICD-10 category level.

    The matrix is (N_categories x N_categories) where rows are ground-truth
    and columns are predicted categories.

    Returns
    -------
    matrix : np.ndarray of shape (N, N)
    labels : list of category names
    """
    # Collect category-level assignments
    cat_gt_per_patient: Dict[str, str] = {}
    cat_pred_per_patient: Dict[str, str] = {}

    for pid, gt_codes in ground_truth.items():
        if pid not in predictions:
            continue
        # Use the first ground-truth code's category
        if gt_codes:
            primary = list(gt_codes)[0]
            cat_gt_per_patient[pid] = get_icd10_category(primary)

        pred_codes = predictions[pid][:top_k]
        if pred_codes:
            cat_pred_per_patient[pid] = get_icd10_category(pred_codes[0])
        else:
            cat_pred_per_patient[pid] = "NONE"

    # Align patients
    pids = sorted(set(cat_gt_per_patient.keys()) & set(cat_pred_per_patient.keys()))

    all_cats = sorted(set(cat_gt_per_patient.values()) | set(cat_pred_per_patient.values()))

    y_true = [cat_gt_per_patient[pid] for pid in pids]
    y_pred = [cat_pred_per_patient[pid] for pid in pids]

    cm = confusion_matrix(y_true, y_pred, labels=all_cats)

    return cm, all_cats


def compute_error_reduction(
    model_accuracy: float,
    baseline_accuracy: float,
    physician_accuracy: float,
) -> Dict[str, float]:
    """Compute error reduction relative to baseline and physician.

    Error reduction = (Baseline_Error - Model_Error) / Baseline_Error * 100
    """
    model_error = 1.0 - model_accuracy
    baseline_error = 1.0 - baseline_accuracy
    physician_error = 1.0 - physician_accuracy

    reduction_vs_baseline = (
        ((baseline_error - model_error) / baseline_error * 100)
        if baseline_error > 0 else 0.0
    )
    reduction_vs_physician = (
        ((physician_error - model_error) / physician_error * 100)
        if physician_error > 0 else 0.0
    )

    return {
        "model_accuracy": round(model_accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "physician_accuracy": round(physician_accuracy, 4),
        "model_error_rate": round(model_error, 4),
        "baseline_error_rate": round(baseline_error, 4),
        "physician_error_rate": round(physician_error, 4),
        "error_reduction_vs_baseline_pct": round(reduction_vs_baseline, 2),
        "error_reduction_vs_physician_pct": round(reduction_vs_physician, 2),
    }


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADRAU-LLM Diagnosis Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True,
        help="Path to ground-truth diagnoses JSONL file.",
    )
    parser.add_argument(
        "--baseline_results", type=str, default=None,
        help="Path to baseline model predictions JSONL (for error reduction).",
    )
    parser.add_argument(
        "--physician_accuracy", type=float, default=None,
        help="Physician Top-1 accuracy (for error reduction comparison).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation/diagnosis_results",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Print results to stdout only; do not save files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load data ---
    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)

    # Align on common patient IDs
    common_pids = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    if not common_pids:
        logger.error(
            "No common patient IDs between predictions and ground truth. "
            "Predictions: %d, Ground truth: %d",
            len(predictions), len(ground_truth),
        )
        sys.exit(1)

    predictions = {pid: predictions[pid] for pid in common_pids}
    ground_truth = {pid: ground_truth[pid] for pid in common_pids}
    logger.info("Evaluating on %d matched records", len(common_pids))

    # --- Compute metrics ---
    top1_acc = compute_topk_accuracy(predictions, ground_truth, k=1)
    top3_acc = compute_topk_accuracy(predictions, ground_truth, k=3)

    per_cat_top1 = compute_per_category_metrics(predictions, ground_truth, top_k=1)
    per_cat_top3 = compute_per_category_metrics(predictions, ground_truth, top_k=3)

    weighted_f1_top1 = compute_weighted_f1(predictions, ground_truth, top_k=1)
    weighted_f1_top3 = compute_weighted_f1(predictions, ground_truth, top_k=3)

    cm, cm_labels = build_confusion_matrix(predictions, ground_truth, top_k=1)

    # --- Error reduction ---
    error_reduction = None
    if args.baseline_results:
        baseline_preds = load_predictions(args.baseline_results)
        baseline_preds = {pid: baseline_preds[pid] for pid in common_pids if pid in baseline_preds}
        baseline_top1 = compute_topk_accuracy(baseline_preds, ground_truth, k=1)
    else:
        baseline_top1 = None

    physician_acc = args.physician_accuracy

    if baseline_top1 is not None and physician_acc is not None:
        error_reduction = compute_error_reduction(top1_acc, baseline_top1, physician_acc)

    # --- Print results ---
    print("=" * 70)
    print("ADRAU-LLM Diagnosis Evaluation Results")
    print("=" * 70)
    print(f"  Records evaluated:           {len(common_pids)}")
    print(f"  Top-1 Accuracy:              {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-3 Accuracy:              {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    print(f"  Weighted F1 (Top-1):         {weighted_f1_top1:.4f}")
    print(f"  Weighted F1 (Top-3):         {weighted_f1_top3:.4f}")
    print()

    print("-" * 70)
    print("Per-Category Metrics (Top-1)")
    print("-" * 70)
    print(f"  {'Category':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
    for cat in sorted(per_cat_top1.keys()):
        m = per_cat_top1[cat]
        print(f"  {cat:<20s} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>8d}")

    if error_reduction:
        print()
        print("-" * 70)
        print("Error Reduction Analysis")
        print("-" * 70)
        er = error_reduction
        print(f"  Model Top-1 Accuracy:        {er['model_accuracy']:.4f}")
        print(f"  Baseline Top-1 Accuracy:     {er['baseline_accuracy']:.4f}")
        print(f"  Physician Top-1 Accuracy:    {er['physician_accuracy']:.4f}")
        print(f"  Error Reduction vs Baseline: {er['error_reduction_vs_baseline_pct']:.1f}%")
        print(f"  Error Reduction vs Physician:{er['error_reduction_vs_physician_pct']:.1f}%")

    print("=" * 70)

    # --- Save outputs ---
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        summary = {
            "num_records": len(common_pids),
            "top1_accuracy": round(top1_acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "weighted_f1_top1": round(weighted_f1_top1, 4),
            "weighted_f1_top3": round(weighted_f1_top3, 4),
            "per_category_top1": per_cat_top1,
            "per_category_top3": per_cat_top3,
        }
        if error_reduction:
            summary["error_reduction"] = error_reduction

        summary_path = out_dir / "diagnosis_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Metrics summary saved to %s", summary_path)

        # Confusion matrix CSV
        cm_path = out_dir / "confusion_matrix.csv"
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ground_truth\\predicted"] + cm_labels)
            for i, row in enumerate(cm):
                writer.writerow([cm_labels[i]] + [int(v) for v in row])
        logger.info("Confusion matrix saved to %s", cm_path)

        # Per-category CSV
        cat_path = out_dir / "per_category_metrics.csv"
        with open(cat_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "precision", "recall", "f1", "support"])
            for cat in sorted(per_cat_top1.keys()):
                m = per_cat_top1[cat]
                writer.writerow([cat, m["precision"], m["recall"], m["f1"], m["support"]])
        logger.info("Per-category metrics saved to %s", cat_path)


if __name__ == "__main__":
    main()
