# ADRAU-LLM: Appropriate Diagnosis and Rational Antibiotic Use Large Language Model

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-ffd21e.svg)](https://huggingface.co/)
<!-- Add additional badges as appropriate (e.g., arXiv, DOI) -->

ADRAU-LLM is a LoRA-fine-tuned, Qwen-based clinical decision support system (CDSS) that integrates respiratory infection diagnosis with antimicrobial stewardship. Trained on 133,450 real-world outpatient electronic health records (EHRs) and evaluated on a temporal hold-out set of 40,559 encounters, the model achieves a Top-1 F1 score of 0.703, representing a 66.2% error reduction compared to the base model. The system provides differential diagnosis and rational antibiotic use recommendations �?addressing two critical, intertwined challenges in primary care.

---

## Table of Contents

- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## Key Results

| Metric | Base Model (Qwen3-8B) | ADRAU-LLM (LoRA) |
|---|---|---|
| Top-1 F1 (Diagnosis) | 0.174 | **0.703** |
| Top-3 F1 (Diagnosis) | 0.331 | **0.832** |
| Error Reduction | �?| **66.2%** |
| Antibiotic Appropriateness | �?| Significantly above baseline |

Additional metrics and ablation studies are available in the accompanying manuscript.

---

## Repository Structure

```
ADRAU-LLM/
├── README.md                 # Project overview and documentation
├── LICENSE                   # CC BY-NC 4.0 license
├── MODEL_CARD.md             # HuggingFace model card
├── CITATION.cff              # Citation metadata (CFF format)
├── requirements.txt          # Python dependencies
├── src/
�?  ├── data/                 # Data preprocessing and loading
�?  ├── model/                # Model definition and LoRA configuration
�?  ├── train/                # Training scripts and configuration
�?  ├── eval/                 # Evaluation and metric computation
�?  └── inference/            # Inference pipeline and utilities
├── configs/                  # YAML configuration files
├── scripts/                  # Utility and helper scripts
├── knowledge_base/           # Antimicrobial stewardship knowledge base
├── prompts/                  # Prompt templates and few-shot examples
└── results/                  # Evaluation outputs and figures
```

---

## Installation

**Requirements:** Python 3.10+, CUDA-compatible GPU (24 GB+ VRAM recommended for full inference).

```bash
git clone https://github.com/Lysue/ADRAU-LLM.git
cd ADRAU-LLM
pip install -r requirements.txt
```

For 4-bit quantized inference (reduced VRAM), install optional dependencies:

```bash
pip install bitsandbytes
```

---

## Quick Start

1. **Clone the repository and install dependencies** (see above).

2. **Download model weights** from [HuggingFace Hub](https://huggingface.co/) (update with actual model ID):

   ```bash
   # Using huggingface-cli
   huggingface-cli download Lysue/ADRAU-LLM --local-dir ./checkpoints/adrau-llm
   ```

3. **Run inference** on a sample case. The merged model loads directly\—no PEFT adapter needed:

   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_id = "./checkpoints/adrau-llm"  # or "Lysue/ADRAU-LLM"

   tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       torch_dtype=torch.bfloat16,
       device_map="auto",
       trust_remote_code=True,
   )

   messages = [
       {"role": "system", "content": "You are a clinical decision support system for respiratory infections."},
       {"role": "user", "content": "Patient: 35F, cough + fever for 5 days, WBC 14.2. Top-3 diagnoses and antibiotic plan?"},
   ]
   text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   inputs = tokenizer(text, return_tensors="pt").to(model.device)

   with torch.no_grad():
       outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2, top_p=0.9)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

---

## Usage

### Basic Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "./checkpoints/adrau-llm"  # or "Lysue/ADRAU-LLM"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a clinical decision support system for respiratory infections."},
    {"role": "user", "content": "Patient: 35F, cough + fever for 5 days, WBC 14.2. Top-3 diagnoses and antibiotic plan?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Batch Evaluation

```bash
python scripts/evaluate.py \
    --model_path ./checkpoints/adrau-llm \
    --test_file ./data/test_set.jsonl \
    --output_dir ./results/ \
    --batch_size 8
```

---

## Data Availability

The electronic health record (EHR) dataset used in this study contains sensitive, de-identified patient information and **cannot be publicly shared** due to institutional ethics committee restrictions and data governance policies.

The following resources are publicly available in this repository:

- **Knowledge base** data for antimicrobial stewardship rules (`knowledge_base/`)
- **Prompt templates** and few-shot examples (`prompts/`)
- **Training, evaluation, and inference source code** (`src/`)
- **Configuration files** (`configs/`)
- **Evaluation results and figures** (`results/`)

---

## Citation

If you use ADRAU-LLM in your research, please cite:

```bibtex
@article{adraullm2025,
  title   = {ADRAU-LLM: Appropriate Diagnosis and Rational Antibiotic Use via Large Language Model Fine-Tuning},
  author  = {},  <!-- update with author list -->
  journal = {},  <!-- update with journal -->
  year    = {2025},
  doi     = {},  <!-- update with DOI -->
}
```

A CITATION.cff file is also provided for GitHub-compatible citation metadata.

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International** (CC BY-NC 4.0) license. See the [LICENSE](LICENSE) file for the full legal text.

In summary: you are free to share and adapt the material for non-commercial purposes, provided you give appropriate credit.
