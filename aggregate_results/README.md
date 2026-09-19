# ADRAU-LLM aggregate public data

This directory contains aggregate, non-identifiable results supporting the
ADRAU-LLM manuscript. It does not contain individual participant data (IPD),
electronic health-record narratives, visit identifiers, timestamps,
row-level predictions, case-level error traces or raw pharmacist-rating rows.

## Files

- `overall_results.csv`: overall diagnosis, antibiotic-use and pharmacist-review
  summaries.
- `bmj_results.csv`: BMJ Never, Sometimes and Always aggregate counts and
  antibiotic-use rates.
- `age_stratified_results.csv`: adult and paediatric aggregate results.
- `pharmacist_review_results.csv`: overall blinded-review strict and flexible
  summaries.

## Interpretation

The 66.2% relative reduction is the comparison of total inappropriate
antibiotic-use decisions between ADRAU-LLM and physician-recorded prescribing
in the evaluable combined BMJ Never and Always groups. It is not an Always-group
comparison. In the Always group, ADRAU-LLM had 328 underuse decisions versus
308 for physicians and 638 for the base model.

Percentages are rounded to one decimal place. Confidence intervals are
record-level percentile-bootstrap intervals with 1,000 iterations where shown
in the manuscript analyses.

