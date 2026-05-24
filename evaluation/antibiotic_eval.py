#!/usr/bin/env python3
"""
ADRAU-LLM Antibiotic Recommendation Evaluation
===============================================

Evaluates antibiotic prescribing recommendations against BMJ Best Practice
guidelines. Computes:

    1. BMJ Criteria Classification (never / sometimes / always antibiotics)
       based on ICD-10 diagnosis codes.
    2. Antibiotic recommendation rates per BMW category.
    3. McNemar test for paired comparisons between model and physician
       (or model and baseline).
    4. Error reduction relative to physician and base model.
    5. Over-prescription and under-prescription rates.

Input Formats
-------------
Predictions JSONL (one JSON object per line):
    {
        "patient_id": "P001",
        "antibiotic": {
            "indicated": true,
            "drug_name": "Amoxicillin"
        },
        "diagnoses": [
            {"rank": 1, "icd10_code": "J15.9", "diagnosis_name": "..."}
        ]
    }

Ground Truth JSONL:
    {
        "patient_id": "P001",
        "icd10_primary": "J15.9",
        "antibiotic_indicated": true,
        "antibiotic_drug": "Amoxicillin",
        "physician_prescribed": true
    }

Usage:
    python antibiotic_eval.py \
        --predictions ./outputs/predictions.jsonl \
        --ground_truth ./data/antibiotic_gold.jsonl \
        --physician_results ./data/physician_prescriptions.jsonl \
        --baseline_results ./baseline/antibiotic_baseline.jsonl \
        --output_dir ./evaluation/antibiotic_results
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
from scipy import stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("adrau-llm.antibiotic_eval")


# ===================================================================
# BMJ Criteria Classification
# ===================================================================

# Categories based on BMJ Best Practice guidelines for antibiotic necessity
# Determined by ICD-10 diagnosis code.

BMJ_ALWAYS_INDICATED = {
    # Bacterial pneumonia
    "J13", "J14", "J15", "J15.0", "J15.1", "J15.2", "J15.3", "J15.4",
    "J15.5", "J15.6", "J15.7", "J15.8", "J15.9",
    # Urinary tract infection
    "N39.0",
    # Sepsis / bacteremia
    "A40", "A40.0", "A40.1", "A40.2", "A40.3", "A40.8", "A40.9",
    "A41", "A41.0", "A41.1", "A41.2", "A41.3", "A41.4", "A41.5",
    "A41.8", "A41.9",
    # Cellulitis / abscess
    "L03", "L03.0", "L03.1", "L03.2", "L03.3", "L03.8", "L03.9",
    "L02", "L02.0", "L02.1", "L02.2", "L02.3", "L02.4", "L02.8", "L02.9",
    # Acute bacterial sinusitis
    "J01", "J01.0", "J01.1", "J01.2", "J01.3", "J01.4", "J01.8", "J01.9",
    # Acute pyelonephritis
    "N10", "N12",
    # Acute appendicitis
    "K35", "K35.2", "K35.3", "K35.8",
    # Suppurative otitis media
    "H66", "H66.0", "H66.3", "H66.4",
    # Acute tonsillitis (bacterial)
    "J03", "J03.0", "J03.8", "J03.9",
    # Bacterial meningitis
    "G00", "G00.0", "G00.1", "G00.2", "G00.3", "G00.8", "G00.9",
    # Cholecystitis
    "K81", "K81.0", "K81.1", "K81.8", "K81.9",
    # Diverticulitis
    "K57", "K57.0", "K57.2", "K57.4", "K57.8",
    # Osteomyelitis
    "M86", "M86.0", "M86.1", "M86.2", "M86.3", "M86.4", "M86.5",
    "M86.6", "M86.8", "M86.9",
    # STI (bacterial)
    "A54", "A54.0", "A54.1", "A54.2", "A54.3", "A54.4", "A54.5",
    "A54.6", "A54.8", "A54.9",
    "A56", "A56.0", "A56.1", "A56.2", "A56.3", "A56.4", "A56.8",
}

BMJ_SOMETIMES_INDICATED = {
    # Acute upper respiratory infection (bacterial superinfection possible)
    "J06", "J06.0", "J06.8", "J06.9",
    # Acute bronchitis
    "J20", "J20.0", "J20.1", "J20.2", "J20.3", "J20.4", "J20.5",
    "J20.6", "J20.7", "J20.8", "J20.9",
    # Acute pharyngitis
    "J02", "J02.0", "J02.8", "J02.9",
    # COPD exacerbation
    "J44.0", "J44.1",
    # Acute exacerbation of asthma
    "J45", "J45.0", "J45.1", "J45.8", "J45.9", "J46",
    # Gastritis / PUD (H. pylori)
    "K29", "K29.0", "K29.1", "K29.2", "K29.3", "K29.4", "K29.5",
    "K29.6", "K29.7", "K29.8", "K29.9",
    "K25", "K25.0", "K25.1", "K25.2", "K25.3", "K25.4", "K25.5",
    "K25.6", "K25.7", "K25.9",
    "K26", "K26.0", "K26.1", "K26.2", "K26.3", "K26.4", "K26.5",
    "K26.6", "K26.7", "K26.9",
    # Fever of unknown origin
    "R50", "R50.0", "R50.2", "R50.8", "R50.9",
    # Myalgia / myositis
    "M79.1",
    # Acute cystitis
    "N30.0", "N30.9",
    # Prostatitis
    "N41", "N41.0", "N41.1", "N41.2", "N41.3", "N41.8", "N41.9",
    # Infectious gastroenteritis
    "A09", "A09.0", "A09.9",
    # Acute otitis media
    "H66.9",
}

BMJ_NEVER_INDICATED = {
    # Common cold / nasopharyngitis
    "J00",
    # Viral infection, unspecified
    "B34", "B34.0", "B34.1", "B34.2", "B34.3", "B34.4", "B34.8", "B34.9",
    # Cough
    "R05",
    # Pain in throat and chest
    "R07", "R07.0", "R07.1", "R07.2", "R07.3", "R07.4",
    # Headache
    "R51",
    # Influenza (uncomplicated)
    "J10", "J10.0", "J10.1", "J10.8",
    "J11", "J11.0", "J11.1", "J11.8",
    # Acute laryngitis / tracheitis (viral)
    "J04", "J04.0", "J04.1", "J04.2",
    # Allergic rhinitis
    "J30", "J30.0", "J30.1", "J30.2", "J30.3", "J30.4",
    # Non-infective gastroenteritis
    "K52", "K52.0", "K52.1", "K52.2", "K52.3", "K52.8", "K52.9",
    # Functional dyspepsia
    "K30",
    # Low back pain
    "M54.5",
    # Essential hypertension
    "I10",
    # Type 2 diabetes (uncomplicated)
    "E11", "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5",
    "E11.6", "E11.7", "E11.8", "E11.9",
    # Anxiety / depression
    "F41", "F41.0", "F41.1", "F41.2", "F41.3", "F41.8", "F41.9",
    "F32", "F32.0", "F32.1", "F32.2", "F32.3", "F32.4", "F32.5",
    "F32.8", "F32.9",
    # Insomnia
    "G47.0",
    # Dermatitis / eczema
    "L20", "L20.0", "L20.8", "L20.9",
    "L30", "L30.0", "L30.1", "L30.2", "L30.3", "L30.4", "L30.5",
    "L30.8", "L30.9",
    # Osteoarthritis
    "M15", "M16", "M17", "M18", "M19",
}


def classify_bmj_category(icd10_code: str) -> str:
    """Classify an ICD-10 code into BMJ antibiotic necessity category.

    Returns one of: 'always', 'sometimes', 'never', 'unknown'.
    """
    code = icd10_code.strip().upper()

    # Exact match
    if code in BMJ_ALWAYS_INDICATED:
        return "always"
    if code in BMJ_SOMETIMES_INDICATED:
        return "sometimes"
    if code in BMJ_NEVER_INDICATED:
        return "never"

    # Try prefix matching (e.g., J15.* -> check base category J15)
    base = code.split(".")[0] if "." in code else code
    for cat_codes, label in [
        (BMJ_ALWAYS_INDICATED, "always"),
        (BMJ_SOMETIMES_INDICATED, "sometimes"),
        (BMJ_NEVER_INDICATED, "never"),
    ]:
        for cat_code in cat_codes:
            cat_base = cat_code.split(".")[0] if "." in cat_code else cat_code
            if base == cat_base:
                return label

    return "unknown"


# ===================================================================
# Data loading
# ===================================================================

def load_predictions(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """Load model predictions from JSONL.

    Returns
    -------
    dict[str, dict]
        patient_id -> {indicated: bool, drug: str, icd10_code: str}
    """
    predictions: Dict[str, Dict[str, Any]] = {}

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

            abx = rec.get("antibiotic", {})
            indicated = abx.get("indicated", False)
            drug = abx.get("drug_name", "")

            # Get primary ICD-10 from top-1 diagnosis
            diagnoses = rec.get("diagnoses", [])
            icd10_code = ""
            if diagnoses:
                icd10_code = diagnoses[0].get("icd10_code", "")

            predictions[pid] = {
                "indicated": bool(indicated),
                "drug": drug,
                "icd10_code": icd10_code,
            }

    logger.info("Loaded %d predictions from %s", len(predictions), jsonl_path)
    return predictions


def load_ground_truth(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """Load ground-truth antibiotic labels from JSONL.

    Returns
    -------
    dict[str, dict]
        patient_id -> {indicated: bool, drug: str, icd10_code: str,
                       physician_prescribed: bool | None}
    """
    ground_truth: Dict[str, Dict[str, Any]] = {}

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

            indicated = rec.get("antibiotic_indicated", None)
            drug = rec.get("antibiotic_drug", "")
            icd10_code = rec.get("icd10_primary", "")
            physician_prescribed = rec.get("physician_prescribed", None)

            ground_truth[pid] = {
                "indicated": bool(indicated) if indicated is not None else None,
                "drug": drug,
                "icd10_code": icd10_code,
                "physician_prescribed": (
                    bool(physician_prescribed)
                    if physician_prescribed is not None else None
                ),
            }

    logger.info("Loaded %d ground-truth records from %s", len(ground_truth), jsonl_path)
    return ground_truth


# ===================================================================
# Metrics
# ===================================================================

def compute_bmj_rates(
    predictions: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute antibiotic recommendation rates stratified by BMJ category."""
    # Align on common patient IDs
    common = sorted(set(predictions.keys()) & set(ground_truth.keys()))

    category_stats: Dict[str, Dict[str, Any]] = {
        "always":    {"total": 0, "model_recommended": 0, "gt_indicated": 0},
        "sometimes": {"total": 0, "model_recommended": 0, "gt_indicated": 0},
        "never":     {"total": 0, "model_recommended": 0, "gt_indicated": 0},
        "unknown":   {"total": 0, "model_recommended": 0, "gt_indicated": 0},
    }

    for pid in common:
        pred = predictions[pid]
        gt = ground_truth[pid]

        icd10 = gt.get("icd10_code", pred.get("icd10_code", ""))
        category = classify_bmj_category(icd10)

        cat = category_stats[category]
        cat["total"] += 1
        if pred["indicated"]:
            cat["model_recommended"] += 1
        if gt.get("indicated", False):
            cat["gt_indicated"] += 1

    # Build summary
    summary: Dict[str, Any] = {"total_evaluated": len(common)}
    for cat in ["always", "sometimes", "never", "unknown"]:
        cs = category_stats[cat]
        total = cs["total"]
        summary[cat] = {
            "count": total,
            "model_recommend_rate": round(cs["model_recommended"] / total, 4) if total else 0.0,
            "gt_indicated_rate": round(cs["gt_indicated"] / total, 4) if total else 0.0,
        }

    # Over-prescription: model says YES but BMJ says NEVER
    # Under-prescription: model says NO but BMJ says ALWAYS
    over_prescribed = 0
    under_prescribed = 0
    total_always = 0
    total_never = 0

    for pid in common:
        pred = predictions[pid]
        gt = ground_truth[pid]
        icd10 = gt.get("icd10_code", pred.get("icd10_code", ""))
        category = classify_bmj_category(icd10)

        if category == "never":
            total_never += 1
            if pred["indicated"]:
                over_prescribed += 1
        elif category == "always":
            total_always += 1
            if not pred["indicated"]:
                under_prescribed += 1

    summary["over_prescription"] = {
        "count": over_prescribed,
        "rate": round(over_prescribed / total_never, 4) if total_never else 0.0,
        "denominator": total_never,
    }
    summary["under_prescription"] = {
        "count": under_prescribed,
        "rate": round(under_prescribed / total_always, 4) if total_always else 0.0,
        "denominator": total_always,
    }

    # Overall recommendation rate
    model_yes = sum(1 for pid in common if predictions[pid]["indicated"])
    gt_yes = sum(1 for pid in common if ground_truth[pid].get("indicated", False))
    summary["overall"] = {
        "model_recommendation_rate": round(model_yes / len(common), 4) if common else 0.0,
        "ground_truth_indicated_rate": round(gt_yes / len(common), 4) if common else 0.0,
    }

    return summary


def mcnemar_test(
    predictions: Dict[str, Dict[str, Any]],
    reference: Dict[str, Dict[str, Any]],
    field: str = "indicated",
) -> Dict[str, Any]:
    """Perform McNemar's test for paired binary outcomes.

    Compares model predictions against a reference (physician or baseline model).

    Returns
    -------
    dict with keys: statistic, p_value, contingency_table, significant
    """
    common = sorted(set(predictions.keys()) & set(reference.keys()))

    # Contingency table:
    #                Reference YES   Reference NO
    # Model YES          a               b
    # Model NO           c               d
    a = b = c = d = 0

    for pid in common:
        model_val = predictions[pid].get(field, False)
        ref_val = reference[pid].get(field, False)

        if isinstance(model_val, str):
            model_val = model_val.upper() == "YES"
        if isinstance(ref_val, str):
            ref_val = ref_val.upper() == "YES"

        model_yes = bool(model_val)
        ref_yes = bool(ref_val)

        if model_yes and ref_yes:
            a += 1
        elif model_yes and not ref_yes:
            b += 1
        elif not model_yes and ref_yes:
            c += 1
        else:
            d += 1

    # McNemar's test uses b and c (discordant pairs)
    table = [[a, b], [c, d]]

    if b + c > 0:
        # With continuity correction
        chi2, p_value = stats.chisquare(
            [b, c], f_exp=[(b + c) / 2, (b + c) / 2]
        )
        # Proper McNemar statistic with Yates correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
        p_value_exact = stats.chi2.sf(statistic, 1)
    else:
        statistic = 0.0
        p_value_exact = 1.0

    return {
        "mcnemar_statistic": round(statistic, 4),
        "mcnemar_p_value": round(p_value_exact, 6),
        "contingency_table": {
            "both_yes": a,
            "model_yes_ref_no": b,
            "model_no_ref_yes": c,
            "both_no": d,
        },
        "significant_at_0.05": p_value_exact < 0.05,
        "discordant_pairs": b + c,
        "total_pairs": len(common),
    }


def compute_error_reduction_antibiotic(
    model_rate: float,
    baseline_rate: float,
    physician_rate: float,
    optimal_rate: float,
    metric_name: str = "recommendation",
) -> Dict[str, float]:
    """Compute error reduction for antibiotic recommendation rates.

    Error is measured as absolute deviation from the optimal (BMJ guideline)
    rate. Error reduction = (baseline_error - model_error) / baseline_error.
    """
    model_error = abs(model_rate - optimal_rate)
    baseline_error = abs(baseline_rate - optimal_rate)
    physician_error = abs(physician_rate - optimal_rate)

    reduction_vs_baseline = (
        ((baseline_error - model_error) / baseline_error * 100)
        if baseline_error > 0 else 0.0
    )
    reduction_vs_physician = (
        ((physician_error - model_error) / physician_error * 100)
        if physician_error > 0 else 0.0
    )

    return {
        "metric": metric_name,
        "optimal_rate": round(optimal_rate, 4),
        "model_rate": round(model_rate, 4),
        "model_error": round(model_error, 4),
        "baseline_rate": round(baseline_rate, 4),
        "baseline_error": round(baseline_error, 4),
        "physician_rate": round(physician_rate, 4),
        "physician_error": round(physician_error, 4),
        "error_reduction_vs_baseline_pct": round(reduction_vs_baseline, 2),
        "error_reduction_vs_physician_pct": round(reduction_vs_physician, 2),
    }


def compute_agreement_metrics(
    predictions: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Compute agreement metrics: accuracy, sensitivity, specificity, PPV, NPV."""
    common = sorted(set(predictions.keys()) & set(ground_truth.keys()))

    y_true = []
    y_pred = []

    for pid in common:
        gt_indicated = ground_truth[pid].get("indicated", False)
        pred_indicated = predictions[pid].get("indicated", False)
        y_true.append(1 if gt_indicated else 0)
        y_pred.append(1 if pred_indicated else 0)

    if not y_true:
        return {"accuracy": 0, "sensitivity": 0, "specificity": 0, "ppv": 0, "npv": 0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "positive_predictive_value": round(ppv, 4),
        "negative_predictive_value": round(npv, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_evaluated": len(y_true),
    }


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADRAU-LLM Antibiotic Recommendation Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to model predictions JSONL file.",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True,
        help="Path to ground-truth JSONL file with BMJ-guideline labels.",
    )
    parser.add_argument(
        "--physician_results", type=str, default=None,
        help="Path to physician prescription records JSONL (for McNemar comparison).",
    )
    parser.add_argument(
        "--baseline_results", type=str, default=None,
        help="Path to baseline model predictions JSONL (for error reduction).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation/antibiotic_results",
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

    # --- BMJ rates ---
    bmj_rates = compute_bmj_rates(predictions, ground_truth)

    # --- Agreement metrics ---
    agreement = compute_agreement_metrics(predictions, ground_truth)

    # --- McNemar test vs physician ---
    mcnemar_vs_physician = None
    if args.physician_results:
        physician_predictions = load_predictions(args.physician_results)
        physician_predictions = {
            pid: physician_predictions[pid]
            for pid in common_pids if pid in physician_predictions
        }
        if physician_predictions:
            mcnemar_vs_physician = mcnemar_test(
                predictions, physician_predictions, field="indicated"
            )

    # --- McNemar test vs baseline ---
    mcnemar_vs_baseline = None
    if args.baseline_results:
        baseline_predictions = load_predictions(args.baseline_results)
        baseline_predictions = {
            pid: baseline_predictions[pid]
            for pid in common_pids if pid in baseline_predictions
        }
        if baseline_predictions:
            mcnemar_vs_baseline = mcnemar_test(
                predictions, baseline_predictions, field="indicated"
            )

    # --- Error reduction ---
    error_reduction = None
    if args.baseline_results and args.physician_results:
        model_rate = bmj_rates["overall"]["model_recommendation_rate"]

        baseline_preds = load_predictions(args.baseline_results)
        baseline_preds = {pid: baseline_preds[pid] for pid in common_pids if pid in baseline_preds}
        baseline_rate = (
            sum(1 for pid in baseline_preds if baseline_preds[pid]["indicated"])
            / max(len(baseline_preds), 1)
        ) if baseline_preds else model_rate

        physician_preds = load_predictions(args.physician_results)
        physician_preds = {pid: physician_preds[pid] for pid in common_pids if pid in physician_preds}
        physician_rate = (
            sum(1 for pid in physician_preds if physician_preds[pid]["indicated"])
            / max(len(physician_preds), 1)
        ) if physician_preds else model_rate

        # Optimal rate = GT indicated rate from BMJ guidelines
        gt_rate = bmj_rates["overall"]["ground_truth_indicated_rate"]

        error_reduction = compute_error_reduction_antibiotic(
            model_rate=model_rate,
            baseline_rate=baseline_rate,
            physician_rate=physician_rate,
            optimal_rate=gt_rate,
        )

    # --- Print results ---
    print("=" * 70)
    print("ADRAU-LLM Antibiotic Recommendation Evaluation Results")
    print("=" * 70)
    print(f"  Records evaluated:                {len(common_pids)}")
    print()

    print("-" * 70)
    print("BMJ Category Breakdown")
    print("-" * 70)
    print(f"  {'Category':<15s} {'Count':>7s} {'Model Rec%':>12s} {'GT Indicated%':>15s}")
    for cat in ["always", "sometimes", "never", "unknown"]:
        b = bmj_rates[cat]
        print(f"  {cat:<15s} {b['count']:>7d} {b['model_recommend_rate']*100:>11.1f}% {b['gt_indicated_rate']*100:>14.1f}%")

    print()
    print(f"  Overall Model Recommendation Rate: {bmj_rates['overall']['model_recommendation_rate']*100:.1f}%")
    print(f"  Ground Truth Indicated Rate:       {bmj_rates['overall']['ground_truth_indicated_rate']*100:.1f}%")
    print()
    print(f"  Over-prescription  (never -> yes): {bmj_rates['over_prescription']['count']}/{bmj_rates['over_prescription']['denominator']} ({bmj_rates['over_prescription']['rate']*100:.1f}%)")
    print(f"  Under-prescription (always -> no): {bmj_rates['under_prescription']['count']}/{bmj_rates['under_prescription']['denominator']} ({bmj_rates['under_prescription']['rate']*100:.1f}%)")

    print()
    print("-" * 70)
    print("Agreement Metrics (vs BMJ Ground Truth)")
    print("-" * 70)
    print(f"  Accuracy:               {agreement['accuracy']*100:.2f}%")
    print(f"  Sensitivity:             {agreement['sensitivity']*100:.2f}%")
    print(f"  Specificity:             {agreement['specificity']*100:.2f}%")
    print(f"  PPV (Precision):         {agreement['positive_predictive_value']*100:.2f}%")
    print(f"  NPV:                     {agreement['negative_predictive_value']*100:.2f}%")
    print(f"  Confusion: TP={agreement['true_positives']} TN={agreement['true_negatives']} FP={agreement['false_positives']} FN={agreement['false_negatives']}")

    if mcnemar_vs_physician:
        print()
        print("-" * 70)
        print("McNemar Test: Model vs Physician")
        print("-" * 70)
        mc = mcnemar_vs_physician
        ct = mc["contingency_table"]
        print(f"  Statistic:              {mc['mcnemar_statistic']:.4f}")
        print(f"  p-value:                {mc['mcnemar_p_value']:.6f}")
        print(f"  Significant (p<0.05):   {mc['significant_at_0.05']}")
        print(f"  Contingency:")
        print(f"    Both YES:             {ct['both_yes']}")
        print(f"    Model YES / Ref NO:   {ct['model_yes_ref_no']}")
        print(f"    Model NO / Ref YES:   {ct['model_no_ref_yes']}")
        print(f"    Both NO:              {ct['both_no']}")

    if mcnemar_vs_baseline:
        print()
        print("-" * 70)
        print("McNemar Test: Model vs Baseline")
        print("-" * 70)
        mc = mcnemar_vs_baseline
        ct = mc["contingency_table"]
        print(f"  Statistic:              {mc['mcnemar_statistic']:.4f}")
        print(f"  p-value:                {mc['mcnemar_p_value']:.6f}")
        print(f"  Significant (p<0.05):   {mc['significant_at_0.05']}")
        print(f"  Contingency:")
        print(f"    Both YES:             {ct['both_yes']}")
        print(f"    Model YES / Base NO:  {ct['model_yes_ref_no']}")
        print(f"    Model NO / Base YES:  {ct['model_no_ref_yes']}")
        print(f"    Both NO:              {ct['both_no']}")

    if error_reduction:
        print()
        print("-" * 70)
        print("Error Reduction Analysis")
        print("-" * 70)
        er = error_reduction
        print(f"  Optimal Rate (BMJ):           {er['optimal_rate']*100:.1f}%")
        print(f"  Model Rate:                   {er['model_rate']*100:.1f}%  (error={er['model_error']*100:.1f}%)")
        print(f"  Baseline Rate:                {er['baseline_rate']*100:.1f}%  (error={er['baseline_error']*100:.1f}%)")
        print(f"  Physician Rate:               {er['physician_rate']*100:.1f}%  (error={er['physician_error']*100:.1f}%)")
        print(f"  Error Reduction vs Baseline:  {er['error_reduction_vs_baseline_pct']:.1f}%")
        print(f"  Error Reduction vs Physician: {er['error_reduction_vs_physician_pct']:.1f}%")

    print("=" * 70)

    # --- Save outputs ---
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "num_records": len(common_pids),
            "bmj_category_breakdown": bmj_rates,
            "agreement_metrics": agreement,
        }
        if mcnemar_vs_physician:
            summary["mcnemar_vs_physician"] = mcnemar_vs_physician
        if mcnemar_vs_baseline:
            summary["mcnemar_vs_baseline"] = mcnemar_vs_baseline
        if error_reduction:
            summary["error_reduction"] = error_reduction

        summary_path = out_dir / "antibiotic_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Metrics summary saved to %s", summary_path)

        # BMJ rates CSV
        rates_path = out_dir / "bmj_category_rates.csv"
        with open(rates_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "count", "model_recommend_rate", "gt_indicated_rate"])
            for cat in ["always", "sometimes", "never", "unknown"]:
                b = bmj_rates[cat]
                writer.writerow([cat, b["count"], b["model_recommend_rate"], b["gt_indicated_rate"]])
        logger.info("BMJ category rates saved to %s", rates_path)

        # Agreement CSV
        agree_path = out_dir / "agreement_metrics.csv"
        with open(agree_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, val in agreement.items():
                writer.writerow([key, val])
        logger.info("Agreement metrics saved to %s", agree_path)


if __name__ == "__main__":
    main()
