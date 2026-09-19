# ADRAU-LLM Core Code

This folder contains the core scripts and LLaMA-Factory configuration used for
ADRAU-LLM, a LoRA fine-tuned Qwen3-8B model for respiratory infection diagnosis
and rational antibiotic recommendation.

## Files

| File | Purpose |
|---|---|
| `qwen3_8b_lora_sft.yaml` | LLaMA-Factory supervised fine-tuning configuration for Qwen3-8B with LoRA. |
| `lab_result_enrichment.py` | Enriches structured laboratory test text with full test names and reference ranges. |
| `dataset_resampling.py` | Performs diagnosis dataset balancing with tiered down-sampling and up-sampling. |
| `knowledge_graph_triple_extraction.py` | Extracts respiratory infection and antibiotic stewardship triples from the knowledge graph. |
| `knowledge_graph_qa_generation.py` | Converts knowledge graph triples into question-answer pairs for antibiotic rational-use training. |
| `diagnosis_top1_top3_evaluation.py` | Evaluates Top-1 and Top-3 diagnostic performance from model prediction files. |
| `diagnosis_confusion_matrix.py` | Generates full and category-level confusion matrices for diagnosis evaluation. |
| `bmj_antibiotic_evaluation.py` | Evaluates antibiotic recommendation rates using BMJ appropriateness categories. |
| `pharmacist_evaluation_analysis.py` | Analyzes blinded clinical pharmacist ratings of antibiotic-use and antibiotic-choice appropriateness. |
| `pharmacist_s_class_exploration.py` | Explores pharmacist assessment results within the BMJ "sometimes appropriate" category. |
| `error_reduction_table.py` | Generates summary tables for diagnostic and antibiotic-prescribing error reduction. |
| `supplementary_analysis/` | Reproducible bootstrap, paired McNemar, age-stratified, and expert-review analyses added during manuscript revision. |

## Supplementary Statistical Analyses

Revision-stage analyses are provided in `supplementary_analysis/`. These
scripts calculate record-level percentile-bootstrap confidence intervals,
paired McNemar comparisons, adult and paediatric strata, category-level
F1-score summaries, and strict or flexible expert-review summaries. See
[`supplementary_analysis/README.md`](supplementary_analysis/README.md) for input
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

