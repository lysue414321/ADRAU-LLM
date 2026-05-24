#!/usr/bin/env python3
"""
ADRAU-LLM Inference Script
==========================

Loads a LoRA adapter fine-tuned on Qwen3-8B and performs inference on
structured EHR JSON input, producing:

    1. Top-3 ranked ICD-10 diagnoses (with codes and rationales)
    2. Antibiotic recommendation (YES/NO + recommended generic drug)

Supports both single-case interactive mode and batch-processing mode.

Example Usage (Single Mode)
----------------------------
    python inference.py \\
        --base_model Qwen/Qwen2.5-8B-Instruct \\
        --lora_adapter ./checkpoints/adrau-llm-lora \\
        --ehr_file ./examples/patient_001.json \\
        --output_file ./outputs/prediction_001.json

Example Usage (Batch Mode)
--------------------------
    python inference.py \\
        --base_model Qwen/Qwen2.5-8B-Instruct \\
        --lora_adapter ./checkpoints/adrau-llm-lora \\
        --batch_input ./data/ehr_batch.jsonl \\
        --batch_output ./outputs/predictions.jsonl

Example EHR JSON Input Format
-----------------------------
{
    "patient_id": "P001",
    "chief_complaint": "Fever and productive cough for 5 days",
    "hpi": "45-year-old male with 5-day history of fever up to 39.2C...",
    "physical_exam": "Crackles at right lung base, RR 22/min...",
    "vitals": "Temp: 39.2C, HR: 98 bpm, BP: 128/82, RR: 22, SpO2: 94%",
    "labs": "WBC: 14.2, CRP: 86, PCT: 2.1, eGFR: 92",
    "medications": "Paracetamol 1g q6h PRN",
    "pmh": "Hypertension, Type 2 DM",
    "age": 45,
    "sex": "M",
    "allergies": "Penicillin - rash"
}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("adrau-llm.inference")


# ===================================================================
# Prompt Templates (inference-side, language-identical to training)
# ===================================================================

DIAGNOSIS_SYSTEM_PROMPT = (
    "You are an expert clinical decision-support AI trained on ICD-10 "
    "diagnostic coding. Given the patient's structured EHR data, provide "
    "the three most likely ICD-10 diagnoses ranked by probability."
)

DIAGNOSIS_USER_TEMPLATE = (
    "Patient EHR Summary:\n"
    "  Chief Complaint: {chief_complaint}\n"
    "  History of Present Illness: {hpi}\n"
    "  Physical Examination: {physical_exam}\n"
    "  Vital Signs: {vitals}\n"
    "  Laboratory Findings: {labs}\n"
    "  Medications: {medications}\n"
    "  Past Medical History: {pmh}\n"
    "\n"
    "Provide the top-3 ranked ICD-10 diagnoses with codes and brief "
    "rationale for each. Use the format:\n"
    "1. ICD10_CODE - DIAGNOSIS_NAME\n"
    "   Rationale: EXPLANATION\n"
    "2. ICD10_CODE - DIAGNOSIS_NAME\n"
    "   Rationale: EXPLANATION\n"
    "3. ICD10_CODE - DIAGNOSIS_NAME\n"
    "   Rationale: EXPLANATION"
)

ANTIBIOTIC_SYSTEM_PROMPT = (
    "You are a clinical antibiotic prescribing assistant. Based on the "
    "confirmed ICD-10 diagnosis and patient context, determine whether "
    "antibiotic therapy is indicated according to BMJ Best Practice "
    "guidelines. If indicated, recommend a specific generic antibiotic."
)

ANTIBIOTIC_USER_TEMPLATE = (
    "Diagnosis: {icd10_code} - {diagnosis_name}\n"
    "Patient Context:\n"
    "  Age: {age}, Sex: {sex}\n"
    "  Allergies: {allergies}\n"
    "  Renal Function (eGFR): {egfr}\n"
    "  Pregnancy Status: {pregnancy}\n"
    "  Severity: {severity}\n"
    "\n"
    "Is antibiotic therapy indicated? Answer YES or NO. "
    "If YES, provide the generic drug name and dosing guidance."
)


# ===================================================================
# Output parsing
# ===================================================================

# Regex to parse a diagnosis line like:
#   1. J15.9 - Bacterial pneumonia, unspecified
DIAGNOSIS_LINE_RE = re.compile(
    r"^\s*(\d+)\s*[.)]\s*([A-Z]\d{2}(?:\.\d{1,4})?)\s*[-–—]\s*(.+?)(?:\s*$)",
    re.MULTILINE,
)

RATIONALE_RE = re.compile(
    r"Rationale\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

ANTIBIOTIC_INDICATED_RE = re.compile(
    r"Antibiotic\s+Indicated\s*:\s*(YES|NO)",
    re.IGNORECASE,
)

ANTIBIOTIC_DRUG_RE = re.compile(
    r"Recommended\s+Drug\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

ANTIBIOTIC_YES_NO_RE = re.compile(
    r"\b(YES|NO)\b",
    re.IGNORECASE,
)


@dataclass
class DiagnosisResult:
    """Parsed diagnosis prediction."""
    rank: int
    icd10_code: str
    diagnosis_name: str
    rationale: str = ""


@dataclass
class AntibioticResult:
    """Parsed antibiotic recommendation."""
    indicated: bool
    drug_name: str
    raw_response: str


@dataclass
class InferenceOutput:
    """Complete inference output for one patient."""
    patient_id: str
    diagnoses: List[DiagnosisResult]
    antibiotic: AntibioticResult
    diagnosis_raw: str
    generation_time_s: float
    model_config: Dict[str, Any]


def parse_diagnosis_response(text: str) -> List[DiagnosisResult]:
    """Parse the model's diagnosis output into structured results.

    Expected format (one per rank):
        1. J15.9 - Bacterial pneumonia, unspecified
           Rationale: Fever, elevated WBC, crackles, high PCT suggest...
    """
    results: List[DiagnosisResult] = []

    # Split into rank-based blocks
    lines = text.strip().split("\n")

    for line in lines:
        match = DIAGNOSIS_LINE_RE.match(line)
        if match:
            rank = int(match.group(1))
            icd10_code = match.group(2).strip()
            diagnosis_name = match.group(3).strip()
            # Remove trailing parenthetical garbage, e.g. " - some extra"
            results.append(DiagnosisResult(
                rank=rank,
                icd10_code=icd10_code,
                diagnosis_name=diagnosis_name,
            ))

    # Extract rationales
    rationales = RATIONALE_RE.findall(text)
    for i, rat in enumerate(rationales):
        if i < len(results):
            results[i].rationale = rat.strip()

    # Ensure only top-3
    results.sort(key=lambda x: x.rank)
    return results[:3]


def parse_antibiotic_response(text: str) -> AntibioticResult:
    """Parse the model's antibiotic recommendation output.

    Expected format:
        Antibiotic Indicated: YES
        Recommended Drug: Amoxicillin
        Dosing Guidance: 500mg TID for 7 days
    """
    indicated_match = ANTIBIOTIC_INDICATED_RE.search(text)
    if indicated_match:
        indicated = indicated_match.group(1).upper() == "YES"
    else:
        # Fallback: scan for YES/NO anywhere
        yes_no = ANTIBIOTIC_YES_NO_RE.findall(text)
        if yes_no:
            indicated = yes_no[0].upper() == "YES"
        else:
            indicated = False

    drug_match = ANTIBIOTIC_DRUG_RE.search(text)
    drug_name = drug_match.group(1).strip() if drug_match else "N/A"

    return AntibioticResult(
        indicated=indicated,
        drug_name=drug_name,
        raw_response=text,
    )


# ===================================================================
# Model loading
# ===================================================================

def load_model_and_adapter(
    base_model_name: str,
    adapter_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, Any]:
    """Load the base model and attach the LoRA adapter for inference.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-8B-Instruct').
    adapter_path : str
        Path to the saved LoRA adapter checkpoint.
    device_map : str
        Device mapping strategy passed to from_pretrained.
    torch_dtype : torch.dtype
        Computation dtype (bfloat16 recommended for Ampere+ GPUs).

    Returns
    -------
    model : PeftModel
        The model with LoRA adapter loaded.
    tokenizer : AutoTokenizer
        The tokenizer matching the base model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("Loading base model: %s", base_model_name)
    logger.info("Loading LoRA adapter from: %s", adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",  # left-padding for batch generation
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map=device_map,
    )

    model.eval()
    logger.info("Model + LoRA adapter loaded successfully.")
    return model, tokenizer


# ===================================================================
# Inference
# ===================================================================

@torch.inference_mode()
def run_diagnosis_inference(
    model: Any,
    tokenizer: Any,
    ehr_record: Dict[str, Any],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
) -> Tuple[str, float]:
    """Run diagnosis inference on a single EHR record."""
    user_prompt = DIAGNOSIS_USER_TEMPLATE.format(
        chief_complaint=ehr_record.get("chief_complaint", ""),
        hpi=ehr_record.get("hpi", ""),
        physical_exam=ehr_record.get("physical_exam", ""),
        vitals=ehr_record.get("vitals", ""),
        labs=ehr_record.get("labs", ""),
        medications=ehr_record.get("medications", ""),
        pmh=ehr_record.get("pmh", ""),
    )

    messages = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    t_start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    t_elapsed = time.perf_counter() - t_start

    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response, t_elapsed


@torch.inference_mode()
def run_antibiotic_inference(
    model: Any,
    tokenizer: Any,
    primary_diagnosis: DiagnosisResult,
    ehr_record: Dict[str, Any],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> Tuple[str, float]:
    """Run antibiotic recommendation inference based on the top diagnosis."""
    user_prompt = ANTIBIOTIC_USER_TEMPLATE.format(
        icd10_code=primary_diagnosis.icd10_code,
        diagnosis_name=primary_diagnosis.diagnosis_name,
        age=ehr_record.get("age", "N/A"),
        sex=ehr_record.get("sex", "N/A"),
        allergies=ehr_record.get("allergies", "None"),
        egfr=ehr_record.get("labs", ""),  # crude fallback; prefer dedicated field
        pregnancy=ehr_record.get("pregnancy", "N/A"),
        severity=ehr_record.get("severity", "moderate"),
    )

    messages = [
        {"role": "system", "content": ANTIBIOTIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    t_start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    t_elapsed = time.perf_counter() - t_start

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response, t_elapsed


def predict_single(
    model: Any,
    tokenizer: Any,
    ehr_record: Dict[str, Any],
    gen_config: Dict[str, Any],
) -> InferenceOutput:
    """Run full inference pipeline (diagnosis + antibiotic) for one EHR record.

    Parameters
    ----------
    model : PeftModel
        The LoRA-adapted model in eval mode.
    tokenizer : AutoTokenizer
    ehr_record : dict
        Structured EHR data (see module docstring for schema).
    gen_config : dict
        Generation parameters (temperature, top_p, max_new_tokens).

    Returns
    -------
    InferenceOutput
    """
    patient_id = ehr_record.get("patient_id", "unknown")

    # Step 1: Diagnosis
    diag_raw, diag_time = run_diagnosis_inference(
        model, tokenizer, ehr_record,
        temperature=gen_config.get("temperature", 0.2),
        top_p=gen_config.get("top_p", 0.9),
        max_new_tokens=gen_config.get("max_new_tokens", 512),
    )
    diagnoses = parse_diagnosis_response(diag_raw)

    # Step 2: Antibiotic (based on top-1 diagnosis if available)
    abx_time = 0.0
    abx_raw = ""
    if diagnoses:
        primary_dx = diagnoses[0]
        abx_raw, abx_time = run_antibiotic_inference(
            model, tokenizer, primary_dx, ehr_record,
            temperature=gen_config.get("temperature", 0.2),
            top_p=gen_config.get("top_p", 0.9),
            max_new_tokens=256,
        )
    antibiotic = parse_antibiotic_response(abx_raw) if abx_raw else AntibioticResult(
        indicated=False, drug_name="N/A", raw_response="",
    )

    return InferenceOutput(
        patient_id=patient_id,
        diagnoses=diagnoses,
        antibiotic=antibiotic,
        diagnosis_raw=diag_raw,
        generation_time_s=diag_time + abx_time,
        model_config=gen_config,
    )


def convert_output_to_dict(output: InferenceOutput) -> Dict[str, Any]:
    """Convert InferenceOutput to a plain dict for JSON serialization."""
    return {
        "patient_id": output.patient_id,
        "diagnoses": [asdict(d) for d in output.diagnoses],
        "antibiotic": asdict(output.antibiotic),
        "diagnosis_raw": output.diagnosis_raw,
        "generation_time_s": round(output.generation_time_s, 3),
    }


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADRAU-LLM Inference: ICD-10 Diagnosis + Antibiotic Recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    parser.add_argument(
        "--base_model", type=str,
        default="Qwen/Qwen2.5-8B-Instruct",
        help="Base model identifier on HuggingFace Hub.",
    )
    parser.add_argument(
        "--lora_adapter", type=str, required=True,
        help="Path to the LoRA adapter checkpoint directory.",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto",
        help="Device mapping for model loading.",
    )

    # Generation
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Nucleus sampling threshold.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum number of tokens to generate.",
    )

    # Input/Output (mutually exclusive modes)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--ehr_file", type=str,
        help="Path to a single EHR JSON file for interactive mode.",
    )
    mode_group.add_argument(
        "--ehr_json", type=str,
        help="Inline EHR JSON string for quick testing.",
    )
    mode_group.add_argument(
        "--batch_input", type=str,
        help="Path to JSONL file with multiple EHR records for batch mode.",
    )

    parser.add_argument(
        "--output_file", type=str,
        help="Path to save output (single JSON for single mode, JSONL for batch).",
    )
    parser.add_argument(
        "--batch_output", type=str,
        help="Path to save batch output as JSONL.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output to stdout.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate PEFT availability
    try:
        from peft import PeftModel  # noqa: F401
    except ImportError:
        logger.error(
            "PEFT is required for inference. Install with: pip install peft"
        )
        sys.exit(1)

    # Load model + adapter
    model, tokenizer = load_model_and_adapter(
        base_model_name=args.base_model,
        adapter_path=args.lora_adapter,
        device_map=args.device_map,
    )

    gen_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    # --- Single mode (file) ---
    if args.ehr_file:
        logger.info("Single-file inference mode: %s", args.ehr_file)
        with open(args.ehr_file, "r", encoding="utf-8") as f:
            ehr_record = json.load(f)

        output = predict_single(model, tokenizer, ehr_record, gen_config)
        result = convert_output_to_dict(output)

        if args.verbose:
            print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("Output saved to %s", out_path)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    # --- Single mode (inline JSON) ---
    elif args.ehr_json:
        logger.info("Inline JSON inference mode")
        ehr_record = json.loads(args.ehr_json)
        output = predict_single(model, tokenizer, ehr_record, gen_config)
        result = convert_output_to_dict(output)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # --- Batch mode ---
    elif args.batch_input:
        batch_path = Path(args.batch_input)
        out_path = Path(args.batch_output) if args.batch_output else batch_path.with_suffix(".predictions.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Batch inference mode: %s -> %s", batch_path, out_path)

        with open(batch_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:

            total = 0
            total_time = 0.0

            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ehr_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping line %d: %s", line_num, exc)
                    continue

                output = predict_single(model, tokenizer, ehr_record, gen_config)
                result = convert_output_to_dict(output)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                total += 1
                total_time += output.generation_time_s

                if total % 10 == 0:
                    logger.info(
                        "Processed %d records (avg %.2f s/record)",
                        total, total_time / total,
                    )

        logger.info(
            "Batch complete: %d records in %.1f s (avg %.2f s/record)",
            total, total_time, total_time / max(total, 1),
        )
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
