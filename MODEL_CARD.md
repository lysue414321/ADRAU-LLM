---
language:
  - en
  - zh
license: cc-by-nc-4.0
tags:
  - medical
  - clinical-decision-support
  - respiratory-infection
  - antibiotic-stewardship
  - lora
  - peft
  - qwen
datasets:
  - real-world-ehr
pipeline_tag: text-generation
---

# ADRAU-LLM: Appropriate Diagnosis and Rational Antibiotic Use LLM

## Model Details

- **Model type:** Causal language model (LoRA-fine-tuned Qwen3-8B, merged and exported as full model)
- **Base model:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Fine-tuning method:** Low-Rank Adaptation (LoRA) with rank=128, alpha=256, applied to all linear layers, then merged into base weights
- **Hardware:** 2× NVIDIA RTX 3090 (24 GB each)
- **Training framework:** HuggingFace Transformers + PEFT + Accelerate
- **Precision:** BF16 mixed-precision training

## Uses

ADRAU-LLM is a clinical decision support system designed for two complementary tasks:

1. **Respiratory infection differential diagnosis** -- Given a patient's chief complaint, history, physical examination findings, and available lab results, the model generates a ranked differential diagnosis for common respiratory infections (e.g., community-acquired pneumonia, acute bronchitis, upper respiratory tract infection, COVID-19, influenza).

2. **Antimicrobial stewardship advisory** -- For each diagnosis, the model recommends whether antibiotics are indicated and, if so, suggests the most appropriate agent, dose, and duration consistent with evidence-based guidelines.

### Intended Use

- Clinical decision support for primary care physicians and respiratory specialists
- Medical education and training
- Research on LLM-based clinical reasoning and antibiotic prescribing

### Out-of-Scope Use

- Autonomous clinical decision-making without physician oversight
- Non-respiratory conditions
- Inpatient or ICU settings (not validated)
- Pediatric or immunocompromised populations (not validated)

## Training Data

| Component | Description | Size |
|---|---|---|
| EHR diagnosis data | Real-world outpatient encounters from Peking University Shenzhen Hospital (January 2023–June 2024), de-identified, with structured chief complaints, histories, physical exams, lab results, and attending physician diagnoses | 133,450 encounters (train); 40,559 encounters (temporal hold-out test) |
| Antibiotic QA pairs | Expert-curated question-answer pairs derived from Chinese clinical guidelines (Guiding Principles for Clinical Application of Antibiotics, 2015) and an infection stewardship knowledge graph, covering antibiotic selection, dosing, contraindications, and special populations for respiratory infections | 3,570 pairs |

**Note:** EHR data cannot be publicly shared due to institutional ethics committee restrictions. Knowledge base data, prompt templates, and source code are available in the repository.

## Evaluation Results

Evaluation performed on a temporal hold-out set of 40,559 outpatient encounters (2024) not seen during training.

| Metric | Base Model (Qwen3-8B) | ADRAU-LLM (LoRA) |
|---|---|---|
| Top-1 Accuracy | 0.303 | 0.697 |
| Top-1 Precision | 0.211 | 0.718 |
| Top-1 Recall | 0.185 | 0.698 |
| Top-1 F1 (weighted) | 0.198 | 0.703 |
| Top-3 F1 (weighted) | 0.317 | 0.763 |
| ΔF1 (LoRA \u2013 Base) | \u2013 | +0.504 |
| Antibiotic Use Rate | 14.0% | 16.3% |
| Antibiotic Error Rate | 15.2% | 13.4% |
| Error Reduction vs Physician | \u2013 | 66.2% |
| Expert-rated Appropriateness (Strict) | \u2013 | 78.1% |
| Expert-rated Appropriateness (Flexible) | \u2013 | 93.1% |

## Limitations

- **Single-center data:** Trained exclusively on EHRs from one tertiary hospital in China; performance may vary across different populations, geographic regions, and healthcare settings.
- **No prospective validation:** All evaluations are retrospective on historical data. Prospective clinical trials have not been conducted.
- **Advisory role only:** The model is intended to assist, not replace, clinical judgment. All outputs should be reviewed by a licensed physician.
- **Limited disease scope:** The model covers common respiratory infections only and was not trained on rare diseases, complex comorbidities, or non-respiratory conditions.
- **Temporal drift:** Clinical guidelines and antibiotic resistance patterns evolve over time; model recommendations may become outdated without periodic retraining.

## Bias, Risks, and Ethical Considerations

- **Demographic bias:** The training population may not be representative of all demographic groups. Performance may differ across age, sex, ethnicity, and socioeconomic strata.
- **Over-reliance risk:** There is a risk that clinicians may over-rely on model outputs without critical evaluation, potentially leading to inappropriate prescribing.
- **Antibiotic resistance:** Incorrect model recommendations could theoretically contribute to antimicrobial resistance if used without physician oversight.
- **Data privacy:** The model was trained on de-identified data and does not memorize or reproduce individual patient records. However, as with all LLMs, the possibility of training data leakage cannot be entirely ruled out.

## How to Use

### Load the model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Lysue/ADRAU-LLM"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Format prompt and generate
messages = [
    {
        "role": "system",
        "content": (
            "You are a clinical decision support system specializing in respiratory "
            "infection diagnosis and antibiotic stewardship. Provide evidence-based, "
            "concise responses with differential diagnosis and treatment recommendations."
        ),
    },
    {
        "role": "user",
        "content": (
            "Chief complaint: Fever (38.6C) and productive cough with green sputum for 4 days.\n"
            "History: 58-year-old male, HTN, no known drug allergies, 20 pack-year smoking.\n"
            "Physical exam: Decreased breath sounds right lower lobe, crackles.\n"
            "Labs: WBC 15.2, CRP 98, PCT 2.1.\n\n"
            "Provide differential diagnosis and antibiotic recommendations."
        ),
    },
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Requirements

```
torch>=2.1.0
transformers>=4.40.0
accelerate>=0.28.0
```

## Citation

```bibtex
@article{adraullm2025,
  title   = {ADRAU-LLM: Appropriate Diagnosis and Rational Antibiotic Use via Large Language Model Fine-Tuning},
  author  = {},  <!-- update with author list -->
  journal = {},  <!-- update with journal -->
  year    = {2025},
  doi     = {},  <!-- update with DOI -->
}
```

## License

This model is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. See the LICENSE file in the repository for full terms.
