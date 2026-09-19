# ADRAU-LLM Antibiotic Knowledge Data

## Overview

This directory contains the antibiotic knowledge question-answer (QA) dataset
used for ADRAU-LLM fine-tuning. The QA pairs cover antibiotic-use decisions,
pathogen matching, drug selection, special populations, contraindications and
other antimicrobial stewardship knowledge.

## Dataset Description

`antibiotic_knowledge_qa.json` contains **3,501 unique QA pairs** corresponding
to the available training-data release:

- **1,898 knowledge-graph-derived QA pairs**;
- **1,603 guideline-derived QA pairs** retained after quality review.

The guideline source initially contained 1,626 generated pairs. Twenty-three
pairs that were not supported by the source text were excluded during quality
review. The released JSON contains no duplicate question-answer pairs.

Each JSON record uses the following fields:

| Field | Description |
|---|---|
| `instruction` | General instruction supplied to the model. |
| `input` | Clinical or antimicrobial stewardship question. |
| `output` | Reference answer used for supervised fine-tuning. |

Aggregate, non-identifiable evaluation results are available in
`../aggregate_results/`.

## Data Sources

The released dataset combines two sources:

1. **Knowledge-graph-derived pairs** generated from an infection and
   antibiotic stewardship knowledge graph using forward and reverse question
   templates.

2. **Guideline-derived pairs** generated from the following clinical guidance
   documents and retained after manual quality review:

   - **Chinese Guiding Principles for Clinical Application of Antibiotics
     (2015 Edition)**, covering antibacterial drug classification, therapeutic
     principles and pathogen-directed treatment.

   - **Appropriate Antibiotic Use for Acute Respiratory Tract Infection in
     Adults**, covering antibiotic prescribing for common adult acute
     respiratory tract infections.

## Generation Pipeline

The QA pairs were prepared through a multi-stage process:

1. **Knowledge graph processing:** Relevant entity-relation-entity triples
   were selected and translated into natural-language questions and answers
   using forward and reverse templates.

2. **Guideline-based generation:** Guideline passages were processed with
   dimension-specific prompts using **Qwen-Plus** to generate candidate QA
   pairs.

3. **Quality review:** Reviewers assessed the guideline-derived pairs for
   source consistency, logical clarity, clinical applicability and
   completeness. Only pairs passing all four criteria were retained.

## Data Availability

The complete released dataset of 3,501 QA pairs is provided in
`antibiotic_knowledge_qa.json`. It contains no electronic health records,
patient identifiers or record-level model predictions.

## License and Use

The QA pairs are intended for research use. Users should verify all clinical
content against current guidelines before clinical application. The repository
licence applies to this release; source guidelines remain subject to their own
terms and conditions.

