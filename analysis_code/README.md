# ADRAU-LLM analysis code

These scripts reproduce analyses added during manuscript revision. They operate
on local case-level CSV or Excel files. No patient-level data are included in
this repository.

Install the analysis dependencies with:

```bash
python -m pip install -r analysis_code/requirements.txt
```

## 1. Paired antibiotic-use analysis

`paired_antibiotic_analysis.py` reports antibiotic-use or recommendation rates
overall and within the BMJ Never, Sometimes, and Always strata. It also reports
paired percentage-point differences, McNemar tests, percentile-bootstrap 95%
confidence intervals, and inappropriate-decision reductions in the combined
Never and Always groups. Optional age input produces adult and paediatric
strata.

```bash
python analysis_code/paired_antibiotic_analysis.py \
  --input local_case_level_results.csv \
  --category-column bmj_category \
  --adrau-column adrau_antibiotic \
  --base-column base_antibiotic \
  --physician-column physician_antibiotic \
  --age-column age \
  --iterations 1000 \
  --seed 20260809 \
  --output-dir local_outputs/antibiotic
```

Required categories are `N`, `S`, and `A`. Binary decisions may be encoded as
0/1, yes/no, true/false, or the corresponding Chinese yes/no values.

The Sometimes-stratum comparison is descriptive. Diagnosis code alone does not
determine whether antibiotic use is appropriate in this group.

## 2. Diagnostic bootstrap and age-stratified analysis

`diagnostic_bootstrap_analysis.py` calculates weighted precision, recall,
F1-score, and accuracy or inclusion at Top-1 and Top-3. Top-3 effective
prediction is defined as the true diagnosis when it appears among the first
three predictions, and otherwise as the first-ranked prediction. The script
also reports paired McNemar tests, record-level bootstrap confidence intervals,
age strata, per-category F1-scores, and the mean and standard deviation of
category-level F1-score differences.

```bash
python analysis_code/diagnostic_bootstrap_analysis.py \
  --input local_diagnosis_results.csv \
  --true-column diagnosis \
  --adrau-columns adrau_1 adrau_2 adrau_3 \
  --base-columns base_1 base_2 base_3 \
  --age-column age \
  --iterations 1000 \
  --seed 20260809 \
  --output-dir local_outputs/diagnosis
```

## 3. Blinded expert-review summary

`expert_review_summary.py` reports strict appropriateness (`score = 2`) and
flexible appropriateness (`score >= 1`) for antibiotic-use decisions and,
where evaluable, antibiotic choice. Results can be grouped by BMJ category,
age, and source.

```bash
python analysis_code/expert_review_summary.py \
  --input local_expert_ratings.xlsx \
  --decision-score-column decision_score \
  --choice-score-column choice_score \
  --category-column bmj_category \
  --age-column age \
  --output local_outputs/expert_summary.csv
```

Do not use model ratings as physician ratings. A model-versus-physician expert
comparison requires independent expert scores for both sources on matched
cases. Use `--source-column` only when the input file contains such ratings.
