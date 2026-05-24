#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for ADRAU-LLM (Antibiotic Decision-support and
Real-time Assessment Utility LLM).

Fine-tunes Qwen3-8B (loaded via Qwen2.5-8B-Instruct backbone) using
Low-Rank Adaptation (LoRA) on a dual-dataset comprising structured
EHR diagnosis records and antibiotic QA pairs.

Reference:
    ADRAU-LLM: A LoRA-Fine-Tuned Large Language Model for Real-Time
    Antibiotic Prescribing and ICD-10 Diagnosis in Primary Healthcare

Key hyperparameters (matching the paper):
    - LoRA: rank=128, alpha=256, dropout=0.05, target_modules="all-linear"
    - Training: lr=1e-4, cosine_with_restarts, warmup_ratio=0.1,
      weight_decay=0.01, batch_size=1, gradient_accumulation=8,
      epochs=2, max_seq_length=4096, bf16
    - Hardware: 2x RTX 3090 24GB

Usage:
    python lora_train.py \
        --model_name Qwen/Qwen2.5-8B-Instruct \
        --diagnosis_data ./data/diagnosis_train.jsonl \
        --antibiotic_data ./data/antibiotic_train.jsonl \
        --output_dir ./checkpoints/adrau-llm-lora \
        --use_unsloth
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_pt_utils import get_model_param_count

# ---------------------------------------------------------------------------
# Try importing PEFT / Unsloth -- graceful fallbacks
# ---------------------------------------------------------------------------
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("adrau-llm.train")


# ===================================================================
# Prompt Templates
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
    "rationale for each."
)

DIAGNOSIS_RESPONSE_TEMPLATE = (
    "1. {icd10_rank1} - {diagnosis_rank1}\n"
    "   Rationale: {rationale_rank1}\n"
    "2. {icd10_rank2} - {diagnosis_rank2}\n"
    "   Rationale: {rationale_rank2}\n"
    "3. {icd10_rank3} - {diagnosis_rank3}\n"
    "   Rationale: {rationale_rank3}"
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

ANTIBIOTIC_RESPONSE_TEMPLATE = (
    "Antibiotic Indicated: {indicated}\n"
    "Recommended Drug: {drug_name}\n"
    "Dosing Guidance: {dosing}"
)

# BMJ criteria categories for antibiotic need
BMJ_ALWAYS_CATEGORIES = [
    "J15",  # Bacterial pneumonia
    "N39.0",  # UTI
    "A41",  # Sepsis
    "L03",  # Cellulitis
    "J01",  # Acute sinusitis (bacterial)
    "N10",  # Acute pyelonephritis
    "K35",  # Acute appendicitis
    "H66",  # Suppurative otitis media
]

BMJ_SOMETIMES_CATEGORIES = [
    "J06",  # Acute upper respiratory infection
    "J20",  # Acute bronchitis
    "J02",  # Acute pharyngitis
    "K29",  # Gastritis
    "R50",  # Fever of unknown origin
    "M79.1",  # Myalgia
]

BMJ_NEVER_CATEGORIES = [
    "J00",  # Acute nasopharyngitis (common cold)
    "B34",  # Viral infection, unspecified
    "R05",  # Cough
    "R07",  # Pain in throat and chest
    "R51",  # Headache
]


# ===================================================================
# Datasets
# ===================================================================

class DiagnosisJSONLDataset(Dataset):
    """Loads EHR diagnosis records from a JSONL file.

    Each line is a JSON object with keys:
        chief_complaint, hpi, physical_exam, vitals, labs, medications,
        pmh, icd10_rank1, diagnosis_rank1, rationale_rank1,
        icd10_rank2, diagnosis_rank2, rationale_rank2,
        icd10_rank3, diagnosis_rank3, rationale_rank3
    """

    def __init__(self, jsonl_path: str, tokenizer, max_seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.records: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self.records.append(record)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON at line %d: %s", line_num, exc)

        logger.info("Loaded %d diagnosis records from %s", len(self.records), jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        user_text = DIAGNOSIS_USER_TEMPLATE.format(
            chief_complaint=rec.get("chief_complaint", ""),
            hpi=rec.get("hpi", ""),
            physical_exam=rec.get("physical_exam", ""),
            vitals=rec.get("vitals", ""),
            labs=rec.get("labs", ""),
            medications=rec.get("medications", ""),
            pmh=rec.get("pmh", ""),
        )

        assistant_text = DIAGNOSIS_RESPONSE_TEMPLATE.format(
            icd10_rank1=rec.get("icd10_rank1", ""),
            diagnosis_rank1=rec.get("diagnosis_rank1", ""),
            rationale_rank1=rec.get("rationale_rank1", ""),
            icd10_rank2=rec.get("icd10_rank2", ""),
            diagnosis_rank2=rec.get("diagnosis_rank2", ""),
            rationale_rank2=rec.get("rationale_rank2", ""),
            icd10_rank3=rec.get("icd10_rank3", ""),
            diagnosis_rank3=rec.get("diagnosis_rank3", ""),
            rationale_rank3=rec.get("rationale_rank3", ""),
        )

        # Build chat-format messages
        messages = [
            {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

        tokenized = self._tokenize_messages(messages)
        return tokenized

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask out the prompt portion (loss only on assistant tokens)
        # Find the assistant turn start in the tokenized sequence.
        assistant_start_marker = "<|im_start|>assistant"
        assistant_marker_ids = self.tokenizer.encode(
            assistant_start_marker, add_special_tokens=False
        )
        # Locate the assistant segment
        labels = input_ids.clone()
        marker_len = len(assistant_marker_ids)
        assistant_start = -1
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i : i + marker_len].tolist() == assistant_marker_ids:
                assistant_start = i + marker_len
                break

        if assistant_start > 0:
            labels[:assistant_start] = -100
        else:
            # Fallback: mask everything before the last 25% (approximate)
            split_point = int(len(input_ids) * 0.75)
            labels[:split_point] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AntibioticJSONLDataset(Dataset):
    """Loads antibiotic QA pairs from a JSONL file.

    Each line is a JSON object with keys:
        icd10_code, diagnosis_name, age, sex, allergies, egfr,
        pregnancy, severity, indicated, drug_name, dosing
    """

    def __init__(self, jsonl_path: str, tokenizer, max_seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.records: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self.records.append(record)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON at line %d: %s", line_num, exc)

        logger.info("Loaded %d antibiotic QA records from %s", len(self.records), jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        user_text = ANTIBIOTIC_USER_TEMPLATE.format(
            icd10_code=rec.get("icd10_code", ""),
            diagnosis_name=rec.get("diagnosis_name", ""),
            age=rec.get("age", ""),
            sex=rec.get("sex", ""),
            allergies=rec.get("allergies", "None"),
            egfr=rec.get("egfr", "N/A"),
            pregnancy=rec.get("pregnancy", "N/A"),
            severity=rec.get("severity", "moderate"),
        )

        assistant_text = ANTIBIOTIC_RESPONSE_TEMPLATE.format(
            indicated=rec.get("indicated", "NO"),
            drug_name=rec.get("drug_name", "N/A"),
            dosing=rec.get("dosing", "N/A"),
        )

        messages = [
            {"role": "system", "content": ANTIBIOTIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

        return self._tokenize_messages(messages)

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        assistant_start_marker = "<|im_start|>assistant"
        assistant_marker_ids = self.tokenizer.encode(
            assistant_start_marker, add_special_tokens=False
        )
        marker_len = len(assistant_marker_ids)
        assistant_start = -1
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i : i + marker_len].tolist() == assistant_marker_ids:
                assistant_start = i + marker_len
                break

        if assistant_start > 0:
            labels[:assistant_start] = -100
        else:
            split_point = int(len(input_ids) * 0.75)
            labels[:split_point] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ===================================================================
# Collator
# ===================================================================

@dataclass
class DataCollatorForChatML:
    """Pads sequences within a batch to the longest entry."""

    tokenizer: Any
    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids = torch.cat([
                f["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=f["input_ids"].dtype),
            ])
            attention_mask = torch.cat([
                f["attention_mask"],
                torch.zeros(pad_len, dtype=f["attention_mask"].dtype),
            ])
            labels = torch.cat([
                f["labels"],
                torch.full((pad_len,), self.label_pad_token_id, dtype=f["labels"].dtype),
            ])

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


# ===================================================================
# Model Loading
# ===================================================================

def load_model_and_tokenizer_unsloth(
    model_name: str,
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
) -> Tuple[Any, Any]:
    """Load model and tokenizer using Unsloth for optimized training."""
    if not UNSLOTH_AVAILABLE:
        raise ImportError(
            "Unsloth is not installed. Install with: "
            "pip install unsloth"
        )

    logger.info("Loading model %s via Unsloth (max_seq_length=%d)", model_name, max_seq_length)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=load_in_4bit,
    )

    # Apply LoRA via Unsloth's built-in method
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Unsloth model loaded. Trainable params: %s",
                _format_params(model))
    return model, tokenizer


def load_model_and_tokenizer_peft(
    model_name: str,
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
    bf16: bool = True,
) -> Tuple[Any, Any]:
    """Load model and tokenizer using HuggingFace PEFT with optional 4-bit QLoRA."""
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is not installed. Install with: pip install peft"
        )

    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("Loading model %s via HF Transformers+PEFT", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    logger.info("PEFT model loaded. Trainable params: %s", _format_params(model))
    return model, tokenizer


def _format_params(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"{trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)"


# ===================================================================
# Training
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADRAU-LLM LoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen2.5-8B-Instruct",
        help="Base model identifier on HuggingFace Hub.",
    )
    parser.add_argument(
        "--use_unsloth", action="store_true",
        help="Use Unsloth for optimized LoRA training.",
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true",
        help="Load model in 4-bit quantization (QLoRA).",
    )

    # Data
    parser.add_argument(
        "--diagnosis_data", type=str, required=True,
        help="Path to JSONL file with diagnosis training records.",
    )
    parser.add_argument(
        "--antibiotic_data", type=str, required=True,
        help="Path to JSONL file with antibiotic QA training records.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=4096,
        help="Maximum sequence length for tokenization.",
    )

    # LoRA
    parser.add_argument(
        "--lora_r", type=int, default=128,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=256,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout rate.",
    )

    # Training
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine_with_restarts",
        choices=["cosine", "cosine_with_restarts", "linear", "constant"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Fraction of steps used for linear warmup.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--per_device_batch_size", type=int, default=1,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--bf16", action="store_true", default=True,
        help="Use bfloat16 mixed precision.",
    )
    parser.add_argument(
        "--seed", type=int, default=3407,
        help="Random seed for reproducibility.",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/adrau-llm-lora",
        help="Directory to save LoRA adapter weights.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Run evaluation every N steps.",
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=3,
        help="Maximum number of saved checkpoints.",
    )

    # Distributed
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training (set by torchrun).",
    )

    return parser.parse_args()


def compute_num_training_steps(
    dataset_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    num_gpus: int = 1,
) -> int:
    """Compute total training steps accounting for distributed setup."""
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
    steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
    return steps_per_epoch * num_epochs


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: int,
    num_cycles: float = 0.5,
) -> Any:
    """Build the learning rate scheduler."""
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "cosine_with_restarts":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        T_0 = max(1, num_training_steps // 4)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=0.0,
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        # constant
        return None


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- Distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        logger.info("Distributed training: rank %d / %d", local_rank, world_size)

    device_count = max(1, torch.cuda.device_count())
    logger.info("CUDA devices available: %d", device_count)

    # --- Load model ---
    if args.use_unsloth and UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_and_tokenizer_unsloth(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        if args.use_unsloth:
            logger.warning("Unsloth not available; falling back to PEFT.")
        model, tokenizer = load_model_and_tokenizer_peft(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            bf16=args.bf16,
        )

    # --- Load datasets ---
    logger.info("Loading training datasets...")
    diag_dataset = DiagnosisJSONLDataset(
        jsonl_path=args.diagnosis_data,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    abx_dataset = AntibioticJSONLDataset(
        jsonl_path=args.antibiotic_data,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    combined_dataset = ConcatDataset([diag_dataset, abx_dataset])
    total_samples = len(combined_dataset)
    logger.info(
        "Combined dataset: %d samples (diagnosis: %d, antibiotic: %d)",
        total_samples, len(diag_dataset), len(abx_dataset),
    )

    # --- Data collator ---
    data_collator = DataCollatorForChatML(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    # --- Training arguments ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        run_name="adrau-llm-lora-finetune",
        save_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        logging_dir=str(output_dir / "logs"),
    )

    # --- Build trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- Compute training steps for logging ---
    num_gpus = max(1, device_count if local_rank == -1 else world_size)
    total_steps = compute_num_training_steps(
        dataset_size=total_samples,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        num_gpus=num_gpus,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info("=" * 60)
    logger.info("ADRAU-LLM LoRA Fine-Tuning Configuration")
    logger.info("=" * 60)
    logger.info("  Model:           %s", args.model_name)
    logger.info("  LoRA rank:       %d", args.lora_r)
    logger.info("  LoRA alpha:      %d", args.lora_alpha)
    logger.info("  LoRA dropout:    %.4f", args.lora_dropout)
    logger.info("  Learning rate:   %.1e", args.learning_rate)
    logger.info("  LR scheduler:    %s", args.lr_scheduler)
    logger.info("  Warmup ratio:    %.2f (%d steps)", args.warmup_ratio, warmup_steps)
    logger.info("  Weight decay:    %.4f", args.weight_decay)
    logger.info("  Per-device BS:   %d", args.per_device_batch_size)
    logger.info("  Grad accum:      %d", args.gradient_accumulation_steps)
    logger.info("  Effective BS:    %d", args.per_device_batch_size * args.gradient_accumulation_steps * num_gpus)
    logger.info("  Epochs:          %d", args.num_epochs)
    logger.info("  Max seq length:  %d", args.max_seq_length)
    logger.info("  BF16:            %s", args.bf16)
    logger.info("  Total steps:     %d", total_steps)
    logger.info("  Dataset size:    %d", total_samples)
    logger.info("  GPUs:            %d", num_gpus)
    logger.info("  Output dir:      %s", output_dir)
    logger.info("=" * 60)

    # --- Train ---
    logger.info("Starting training...")
    train_result = trainer.train()

    # --- Save final model ---
    logger.info("Saving LoRA adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # --- Save training metrics ---
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # --- Save LoRA config alongside checkpoint ---
    lora_save_path = output_dir / "adapter_config.json"
    logger.info("LoRA adapter saved. Total training loss: %.4f",
                metrics.get("train_loss", float("nan")))

    logger.info("Fine-tuning complete. Adapter saved to %s", output_dir)


if __name__ == "__main__":
    main()
