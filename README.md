# ADRAU-LLM Code and Aggregate Results

This folder contains the core scripts and LLaMA-Factory configuration used for
ADRAU-LLM, a LoRA fine-tuned Qwen3-8B model for respiratory infection diagnosis
and rational antibiotic recommendation.

## Repository contents

| File | Purpose |
|---|---|
| `analysis_code/` | All preprocessing, fine-tuning support, diagnostic evaluation, antibiotic-use evaluation, pharmacist-review, bootstrap, paired-comparison and age-stratified analysis scripts. |
| `qwen3_8b_lora_sft.yaml` | LLaMA-Factory supervised fine-tuning configuration for Qwen3-8B with LoRA. |
| `data/` | Publicly shareable generated antimicrobial-stewardship QA data and data documentation. |
| `aggregate_results/` | Aggregate, non-identifiable results used to summarize the study findings. |

## Analysis code

All analysis scripts are provided in `analysis_code/`. They calculate
record-level percentile-bootstrap confidence intervals,
paired McNemar comparisons, adult and paediatric strata, category-level
F1-score summaries, and strict or flexible expert-review summaries. See
[`analysis_code/README.md`](analysis_code/README.md) for input
schemas and command examples.

## Fine-Tuning With LLaMA-Factory

The LoRA fine-tuning configuration is provided in:

```text
qwen3_8b_lora_sft.yaml
```

The key settings are:

- Base model: `Qwen/Qwen3-8B`
- Fine-tuning method: LoRA
- LoRA rank: `128`
- LoRA alpha: `256`
- LoRA dropout: `0.05`
- Learning rate: `1.0e-4`
- Epochs: `2`
- Maximum sequence length: `4096`
- Mixed precision: `bf16`
- Generation settings: `temperature=0.2`, `top_p=0.9`

Example command:

```bash
llamafactory-cli train qwen3_8b_lora_sft.yaml
```

Before running, update the dataset name in the YAML file to match the dataset
entry registered in your local LLaMA-Factory `dataset_info.json`.

## Data Privacy

The real outpatient EHR data used for diagnosis training and validation are not
included in this repository because they contain patient-level clinical
narratives. Do not upload files containing outpatient numbers, visit dates,
chief complaints, present illness, physical examination text, laboratory
narratives, or model prediction traces that reconstruct patient cases.

The public release is limited to the following materials:

- generated antibiotic rational-use QA pairs;
- aggregate evaluation results;
- figures derived from aggregate results;
- preprocessing, fine-tuning, and evaluation code.

No individual participant data, EHR narratives, outpatient or visit identifiers,
row-level predictions, case-level error traces, or raw pharmacist-rating rows are
included. The aggregate release is also mirrored in the Open Science Framework
repository listed in the manuscript Data Availability statement once its
persistent record is available.

## Suggested Citation

Using a fine-tuned large language model to assist physician diagnosis and
antibiotic rational use for respiratory tract infections.

