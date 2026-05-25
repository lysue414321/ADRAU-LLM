# ADRAU-LLM Antibiotic QA Pairsantibiotic_knowledge_qa

## Overview

This directory contains question-answer (QA) pairs for fine-tuning large language models on antibiotic clinical knowledge. The QA pairs are in Chinese and cover multiple clinical dimensions relevant to antibiotic prescribing, pathogen matching, special populations, and contraindications.

## Dataset Description

The full dataset contains **3,570 QA pairs** spanning the following categories:

| Category | Description | Approximate Count |
|---|---|---|
| `disease_treatment` | Disease-specific antibiotic treatment recommendations, dosing, and duration | ~1,100 |
| `pathogen_matching` | Pathogen identification, drug-pathogen matching, and resistance-aware treatment selection | ~980 |
| `special_populations` | Dosing and drug selection for pediatrics, geriatrics, pregnancy, renal/hepatic impairment | ~850 |
| `contraindications` | Drug contraindications, adverse reactions, allergy cross-reactivity, and drug-drug interactions | ~640 |

## Data Sources

The QA pairs are derived from two authoritative Chinese clinical guidelines:

1. **Chinese Guiding Principles for Clinical Application of Antibiotics (2015 Edition)** -- The primary national guideline issued by the National Health Commission of China, covering antibacterial drug classification, therapeutic principles, and pathogen-directed therapy across all major infectious disease categories.

2. **Guidelines for Antibiotic Use in Adult Acute Respiratory Infections** -- A focused guideline covering antibiotic prescribing for upper and lower respiratory tract infections in adult patients.

## Generation Pipeline

The QA pairs were generated through a multi-stage process:

1. **Knowledge Graph Construction**: Guideline text was parsed into a structured knowledge graph consisting of entity-relation-entity triples (e.g., `<Community-acquired Pneumonia> -- <first_line_treatment> -- <Amoxicillin>`).

2. **LLM-based Generation**: The knowledge graph triples and guideline passages were fed to **Qwen-Plus** with dimension-specific prompts to generate question-answer pairs in clinical guideline style.


## Accessing the Full Dataset

The complete 3,570 QA pairs are available upon request. Please contact the repository maintainers for access.

## License

The QA pairs are derived from publicly available clinical guidelines and are intended for research purposes. Users should verify clinical content against current guidelines before clinical application.
