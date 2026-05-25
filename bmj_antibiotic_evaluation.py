import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as sp_stats
from statsmodels.stats.contingency_tables import mcnemar

# ─────────────────────────────────────────────────────────
# Font and color config
# ─────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

BLUE_COLORS = {
    'chart_colors': {
        'Reasonable': '#2196F3',
        'Unreasonable': '#1565C0',
        'Yes': '#1976D2',
        'No': '#90CAF9',
        'Unknown': '#B0BEC5'
    },
    'category_colors': ['#E3F2FD', '#BBDEFB', '#64B5F6', '#1976D2', '#1565C0', '#0D47A1'],
    'backgrounds': '#FAFAFA'
}

# Colors for model vs physician model
COMPARISON_COLORS = {
    'Model':  '#1976D2',   # blue  — model
    'Physician':   '#E53935',   # red — physicians
    'S_use_rate': '#78909C',   # blue-gray accent
}


# ═════════════════════════════════════════════════════════
# Class 1: Single-source evaluator (reused for both models)
# ═════════════════════════════════════════════════════════
class AntibioticEvaluator:
    def __init__(self, source_label="Source"):
        """
        source_label: A display name, e.g. "Model" or "Physician"
        """
        self.source_label = source_label
        self.standard_df = None
        self.prediction_df = None
        self.evaluation_results = None

    def load_standard_file(self, file_path):
        try:
            self.standard_df = pd.read_excel(file_path)
            print(f"✅ [{self.source_label}] Loaded standard file: {len(self.standard_df)} records")
            return True
        except Exception as e:
            print(f"❌ [{self.source_label}] Failed to load standard file: {e}")
            return False

    def load_prediction_file(self, file_path):
        try:
            self.prediction_df = pd.read_excel(file_path)
            print(f"✅ [{self.source_label}] Loaded prediction file: {len(self.prediction_df)} records")

            self.prediction_df['extracted_icd10'] = self.prediction_df['pred_icd10_code'].apply(
                self._extract_icd10_code
            )
            self.prediction_df['antibiotic_used'] = self.prediction_df['pred_use_antibiotic'].apply(
                self._standardize_antibiotic_field
            )
            return True
        except Exception as e:
            print(f"❌ [{self.source_label}] Failed to load prediction file: {e}")
            return False

    def _extract_icd10_code(self, code_text):
        if pd.isna(code_text):
            return None
        match = re.search(r'([A-Z]\d{2}\.?\d?)', str(code_text))
        if match:
            return match.group(1)
        return str(code_text).strip()

    def _standardize_antibiotic_field(self, value):
        if pd.isna(value):
            return "Unknown"
        value_str = str(value).strip().lower()
        if value_str in ['是', 'yes', 'y', '1', 'true']:
            return "Yes"
        elif value_str in ['否', 'no', 'n', '0', 'false']:
            return "No"
        return "Unknown"

    def _get_category_meaning(self, category):
        return {'N': 'Never use antibiotics',
                'S': 'Sometimes use antibiotics',
                'A': 'Always use antibiotics'}.get(category, 'Unknown category')

    def evaluate_antibiotic_usage(self):
        if self.standard_df is None or self.prediction_df is None:
            print(f"❌ [{self.source_label}] Load files first")
            return False

        merged = self.prediction_df.merge(
            self.standard_df, left_on='extracted_icd10',
            right_on='ICD10_CODE', how='left'
        )

        def eval_row(row):
            cat, used = row['CATEGORY'], row['antibiotic_used']
            if pd.isna(cat):
                return "No standard", "ICD10 not found"
            if used == "Unknown":
                return "Cannot evaluate", "Antibiotic usage unknown"
            if cat == 'S':
                return "S-UseRate", "S category: analyzed via use rate"
            if cat == 'N':
                return ("Reasonable", "N compliant") if used == "No" else ("Unreasonable", "N violated")
            if cat == 'A':
                return ("Reasonable", "A compliant") if used == "Yes" else ("Unreasonable", "A violated")
            return "Cannot evaluate", f"Unknown cat: {cat}"

        res = merged.apply(eval_row, axis=1, result_type='expand')
        merged['evaluation_result'] = res[0]
        merged['evaluation_reason'] = res[1]
        self.evaluation_results = merged
        return True

    # ───────── Core metric extraction (used by comparison layer) ─────────
    def get_metrics(self):
        """
        Returns a dict of all key metrics for this source.
        Used by ComparativeEvaluator to build side-by-side tables.
        """
        if self.evaluation_results is None:
            return None

        df = self.evaluation_results
        total = len(df)
        metrics = {'source': self.source_label, 'total_cases': total}

        # Per-category appropriateness (A and N)
        for cat in ['N', 'A']:
            cat_data = df[(df['CATEGORY'] == cat) &
                          (df['evaluation_result'].isin(['Reasonable', 'Unreasonable']))]
            n_total = len(cat_data)
            n_reasonable = len(cat_data[cat_data['evaluation_result'] == 'Reasonable'])
            rate = (n_reasonable / n_total * 100) if n_total > 0 else np.nan
            metrics[f'{cat}_total'] = n_total
            metrics[f'{cat}_reasonable'] = n_reasonable
            metrics[f'{cat}_appropriateness_rate'] = rate

        # S category use rate
        s_clear = df[(df['CATEGORY'] == 'S') & (df['antibiotic_used'].isin(['Yes', 'No']))]
        s_used = len(s_clear[s_clear['antibiotic_used'] == 'Yes'])
        s_total = len(s_clear)
        metrics['S_total'] = s_total
        metrics['S_used'] = s_used
        metrics['S_use_rate'] = (s_used / s_total * 100) if s_total > 0 else np.nan

        # Overall rates (denominator = full dataset)
        reasonable_total = len(df[df['evaluation_result'] == 'Reasonable'])
        unreasonable_total = len(df[df['evaluation_result'] == 'Unreasonable'])
        s_flag_total = len(df[df['evaluation_result'] == 'S-UseRate'])
        metrics['reasonable_total'] = reasonable_total
        metrics['unreasonable_total'] = unreasonable_total
        metrics['s_flag_total'] = s_flag_total
        metrics['overall_appropriateness_rate'] = reasonable_total / total * 100 if total else np.nan
        metrics['overall_inappropriate_rate']   = unreasonable_total / total * 100 if total else np.nan
        metrics['overall_s_proportion']         = s_flag_total / total * 100 if total else np.nan

        # Antibiotic use distribution (overall)
        yes_count = len(df[df['antibiotic_used'] == 'Yes'])
        no_count  = len(df[df['antibiotic_used'] == 'No'])
        metrics['yes_count'] = yes_count
        metrics['no_count'] = no_count
        metrics['overall_use_rate'] = (yes_count / (yes_count + no_count) * 100) \
                                       if (yes_count + no_count) > 0 else np.nan

        return metrics


# ═════════════════════════════════════════════════════════
# Class 2: Comparative evaluator (model vs physician)
# ═════════════════════════════════════════════════════════
class ComparativeEvaluator:
    def __init__(self):
        self.model_eval = AntibioticEvaluator(source_label="Model")
        self.physician_eval  = AntibioticEvaluator(source_label="Physician")
        self.model_metrics = None
        self.physician_metrics  = None

    def run(self, standard_file, model_file, physician_file):
        """Load all files, evaluate, compute metrics."""
        print("\n" + "═" * 70)
        print("  Loading and evaluating Model model")
        print("═" * 70)
        ok1 = (self.model_eval.load_standard_file(standard_file) and
               self.model_eval.load_prediction_file(model_file) and
               self.model_eval.evaluate_antibiotic_usage())

        print("\n" + "═" * 70)
        print("  Loading and evaluating Physician model")
        print("═" * 70)
        ok2 = (self.physician_eval.load_standard_file(standard_file) and
               self.physician_eval.load_prediction_file(physician_file) and
               self.physician_eval.evaluate_antibiotic_usage())

        if not (ok1 and ok2):
            print("❌ Evaluation failed")
            return False

        self.model_metrics = self.model_eval.get_metrics()
        self.physician_metrics  = self.physician_eval.get_metrics()
        return True

    # ───────── Paired statistical tests ─────────
    def run_paired_tests(self):
        """
        Run paired McNemar tests for Model vs Physician on the SAME cases.
        Required because both models made decisions on the same patient set.

        Three tests:
        1. N category appropriateness: correct/incorrect → McNemar
        2. A category appropriateness: correct/incorrect → McNemar
        3. S category antibiotic use: Yes/No → McNemar (tests whether the
           two models have different prescribing tendencies)
        Also: overall decision agreement (Yes/No) → McNemar + Cohen's κ
        """
        m_df = self.model_eval.evaluation_results
        p_df = self.physician_eval.evaluation_results
        if m_df is None or p_df is None:
            print("❌ Run evaluation first")
            return None

        # Align by row index — assumes both files contain the SAME cases in the SAME order.
        # This is essential for paired testing.
        n_min = min(len(m_df), len(p_df))
        if len(m_df) != len(p_df):
            print(f"⚠️  Warning: Model has {len(m_df)} rows, Physician has {len(p_df)} rows.")
            print(f"    Truncating both to first {n_min} rows and pairing by row index.")
            print(f"    If files are NOT in the same order, results will be invalid.")

        m_aligned = m_df.iloc[:n_min].reset_index(drop=True)
        p_aligned = p_df.iloc[:n_min].reset_index(drop=True)

        results = {'n_paired': n_min, 'tests': {}}

        # ─── Test 1 & 2: N and A category appropriateness (correct vs incorrect) ───
        for cat in ['N', 'A']:
            # Keep only rows where both sides are in the same BMJ category AND both evaluable
            mask = (
                (m_aligned['CATEGORY'] == cat) &
                (p_aligned['CATEGORY'] == cat) &
                (m_aligned['evaluation_result'].isin(['Reasonable', 'Unreasonable'])) &
                (p_aligned['evaluation_result'].isin(['Reasonable', 'Unreasonable']))
            )
            sub_m = m_aligned[mask]
            sub_p = p_aligned[mask]

            if len(sub_m) < 1:
                results['tests'][f'{cat}_appropriateness'] = {
                    'n': 0, 'note': f'No paired {cat} cases'
                }
                continue

            # 2x2 contingency table:
            #                          Physician Correct | Physician Incorrect
            # Model Correct              a         |         b
            # Model Incorrect            c         |         d
            m_correct = (sub_m['evaluation_result'] == 'Reasonable').values
            p_correct = (sub_p['evaluation_result'] == 'Reasonable').values

            a = int(((m_correct == True) & (p_correct == True)).sum())
            b = int(((m_correct == True) & (p_correct == False)).sum())
            c = int(((m_correct == False) & (p_correct == True)).sum())
            d = int(((m_correct == False) & (p_correct == False)).sum())

            table = np.array([[a, b], [c, d]])
            # Discordant cells: b (model right, physician wrong) and c (model wrong, physician right)
            discordant = b + c

            # Choose McNemar variant based on discordant count
            if discordant < 25:
                # Exact binomial test — recommended for small discordant counts
                test_result = mcnemar(table, exact=True)
                test_type = "McNemar exact (binomial)"
            else:
                # Continuity-corrected chi-square — recommended for larger counts
                test_result = mcnemar(table, exact=False, correction=True)
                test_type = "McNemar χ² (continuity-corrected)"

            stat = test_result.statistic
            pval = test_result.pvalue

            m_rate = m_correct.mean() * 100
            p_rate = p_correct.mean() * 100

            results['tests'][f'{cat}_appropriateness'] = {
                'n': len(sub_m),
                'table': {'both_correct': a,
                          'model_correct_physician_wrong': b,
                          'model_wrong_physician_correct': c,
                          'both_wrong': d},
                'model_rate': m_rate,
                'physician_rate': p_rate,
                'delta': m_rate - p_rate,
                'test_type': test_type,
                'statistic': stat,
                'pvalue': pval,
                'discordant': discordant,
            }

        # ─── Test 3: S category antibiotic use (Yes/No) ───
        mask_s = (
            (m_aligned['CATEGORY'] == 'S') &
            (p_aligned['CATEGORY'] == 'S') &
            (m_aligned['antibiotic_used'].isin(['Yes', 'No'])) &
            (p_aligned['antibiotic_used'].isin(['Yes', 'No']))
        )
        sub_m_s = m_aligned[mask_s]
        sub_p_s = p_aligned[mask_s]

        if len(sub_m_s) >= 1:
            m_yes = (sub_m_s['antibiotic_used'] == 'Yes').values
            p_yes = (sub_p_s['antibiotic_used'] == 'Yes').values

            a = int(((m_yes == True)  & (p_yes == True)).sum())
            b = int(((m_yes == True)  & (p_yes == False)).sum())
            c = int(((m_yes == False) & (p_yes == True)).sum())
            d = int(((m_yes == False) & (p_yes == False)).sum())

            table = np.array([[a, b], [c, d]])
            discordant = b + c
            if discordant < 25:
                tr = mcnemar(table, exact=True)
                test_type = "McNemar exact (binomial)"
            else:
                tr = mcnemar(table, exact=False, correction=True)
                test_type = "McNemar χ² (continuity-corrected)"

            m_use_rate = m_yes.mean() * 100
            p_use_rate = p_yes.mean() * 100

            results['tests']['S_use_rate'] = {
                'n': len(sub_m_s),
                'table': {'both_yes': a,
                          'model_yes_physician_no': b,
                          'model_no_physician_yes': c,
                          'both_no': d},
                'model_rate': m_use_rate,
                'physician_rate': p_use_rate,
                'delta': m_use_rate - p_use_rate,
                'test_type': test_type,
                'statistic': tr.statistic,
                'pvalue': tr.pvalue,
                'discordant': discordant,
            }
        else:
            results['tests']['S_use_rate'] = {'n': 0, 'note': 'No paired S cases'}

        # ─── Test 4: Overall Yes/No decision agreement ───
        mask_all = (
            m_aligned['antibiotic_used'].isin(['Yes', 'No']) &
            p_aligned['antibiotic_used'].isin(['Yes', 'No'])
        )
        sub_m_all = m_aligned[mask_all]
        sub_p_all = p_aligned[mask_all]
        if len(sub_m_all) >= 1:
            m_yes = (sub_m_all['antibiotic_used'] == 'Yes').values
            p_yes = (sub_p_all['antibiotic_used'] == 'Yes').values
            a = int(((m_yes == True)  & (p_yes == True)).sum())
            b = int(((m_yes == True)  & (p_yes == False)).sum())
            c = int(((m_yes == False) & (p_yes == True)).sum())
            d = int(((m_yes == False) & (p_yes == False)).sum())
            table = np.array([[a, b], [c, d]])
            discordant = b + c
            if discordant < 25:
                tr = mcnemar(table, exact=True)
                test_type = "McNemar exact (binomial)"
            else:
                tr = mcnemar(table, exact=False, correction=True)
                test_type = "McNemar χ² (continuity-corrected)"

            # Cohen's κ
            total = a + b + c + d
            po = (a + d) / total if total else 0
            p_m_yes = (a + b) / total if total else 0
            p_p_yes = (a + c) / total if total else 0
            pe = p_m_yes * p_p_yes + (1 - p_m_yes) * (1 - p_p_yes)
            kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan

            results['tests']['overall_use_decision'] = {
                'n': len(sub_m_all),
                'table': {'both_yes': a,
                          'model_yes_physician_no': b,
                          'model_no_physician_yes': c,
                          'both_no': d},
                'model_use_rate': m_yes.mean() * 100,
                'physician_use_rate': p_yes.mean() * 100,
                'agreement_rate': po * 100,
                'kappa': kappa,
                'test_type': test_type,
                'statistic': tr.statistic,
                'pvalue': tr.pvalue,
                'discordant': discordant,
            }

        self.paired_results = results
        return results

    def print_paired_tests(self):
        """Pretty-print paired test results."""
        if not hasattr(self, 'paired_results') or self.paired_results is None:
            self.run_paired_tests()
        r = self.paired_results

        print("\n" + "═" * 80)
        print("🔬 PAIRED STATISTICAL TESTS — Model vs Physician (same cases)")
        print("═" * 80)
        print(f"   Paired cases: n = {r['n_paired']}")
        print(f"   Test: McNemar (paired binomial comparison)")
        print(f"   Cases aligned by row index — files must be in same case order.\n")

        def _sig(p):
            if np.isnan(p): return ""
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return "ns"

        for test_name, display_name in [
            ('N_appropriateness',   'Test 1: N-category appropriateness (correct vs incorrect)'),
            ('A_appropriateness',   'Test 2: A-category appropriateness (correct vs incorrect)'),
            ('S_use_rate',          'Test 3: S-category antibiotic use rate (Yes vs No)'),
            ('overall_use_decision','Test 4: Overall antibiotic use decision (Yes vs No)'),
        ]:
            t = r['tests'].get(test_name, {})
            print("─" * 80)
            print(f"  {display_name}")
            print("─" * 80)
            if t.get('n', 0) == 0:
                print(f"  ⚠️  {t.get('note', 'insufficient data')}")
                continue

            tbl = t['table']
            print(f"  n paired = {t['n']}")
            print(f"  2×2 contingency table:")
            if 'both_correct' in tbl:
                print(f"                               Physician Correct | Physician Incorrect")
                print(f"     Model Correct             {tbl['both_correct']:>5}    |      {tbl['model_correct_physician_wrong']:>5}")
                print(f"     Model Incorrect           {tbl['model_wrong_physician_correct']:>5}    |      {tbl['both_wrong']:>5}")
                print(f"  Model correct rate: {t['model_rate']:.2f}%")
                print(f"  Physician   correct rate: {t['physician_rate']:.2f}%")
            else:
                print(f"                               Physician Yes  | Physician No")
                print(f"     Model Yes                 {tbl['both_yes']:>5}    |    {tbl['model_yes_physician_no']:>5}")
                print(f"     Model No                  {tbl['model_no_physician_yes']:>5}    |    {tbl['both_no']:>5}")
                print(f"  Model use rate:     {t.get('model_rate', t.get('model_use_rate', np.nan)):.2f}%")
                print(f"  Physician   use rate:     {t.get('physician_rate',  t.get('physician_use_rate',  np.nan)):.2f}%")

            delta_val = t['delta'] if 'delta' in t else (t['model_use_rate'] - t['physician_use_rate'])
            print(f"  Δ (Model − Physician): {delta_val:+.2f} pp")
            print(f"  Discordant pairs:          {t['discordant']}")
            print(f"  Test:                      {t['test_type']}")
            print(f"  Statistic:                 {t['statistic']:.4f}")
            print(f"  p-value:                   {t['pvalue']:.4f}  {_sig(t['pvalue'])}")
            if 'kappa' in t:
                print(f"  Cohen's κ:                 {t['kappa']:.3f}  "
                      f"({'poor' if t['kappa']<0.2 else 'fair' if t['kappa']<0.4 else 'moderate' if t['kappa']<0.6 else 'substantial' if t['kappa']<0.8 else 'almost perfect'})")
            print()

        print("─" * 80)
        print("  Significance: *** p<0.001 | ** p<0.01 | * p<0.05 | ns not significant")
        print("─" * 80)

    # ═════════════════════════════════════════════════════════════════
    # ★ Error Analysis — BMJ-discordant decisions in N and A categories
    # ═════════════════════════════════════════════════════════════════
    def run_error_analysis(self):
        """
        Count absolute BMJ-discordant (incorrect) decisions for both sources
        in N and A categories, and compute error reduction of Model
        relative to Physician.

        Error definition:
          - N (never appropriate) case: prescribing antibiotics = error
          - A (always appropriate) case: withholding antibiotics = error
          - S cases excluded (not adjudicable from ICD-10 alone)

        Reduction formula:
          reduction_% = (E_physician − E_model) / E_physician × 100
        """
        m_df = self.model_eval.evaluation_results
        p_df = self.physician_eval.evaluation_results
        if m_df is None or p_df is None:
            print("❌ Run evaluation first")
            return None

        n_min = min(len(m_df), len(p_df))
        m_data = m_df.iloc[:n_min].reset_index(drop=True)
        p_data = p_df.iloc[:n_min].reset_index(drop=True)

        cmp_label = "Physician"

        print("\n" + "═" * 80)
        print(f"🔬 ERROR ANALYSIS — Model vs {cmp_label} (BMJ N + A only)")
        print("═" * 80)
        print(f"   Error = BMJ-discordant decision")
        print(f"   (antibiotic prescribed in N case, or withheld in A case)")
        print(f"   Reduction = (E_{cmp_label.lower()} − E_model) / E_{cmp_label.lower()} × 100%\n")

        header = (f"   {'Category':<12}{'n paired':>10}"
                  f"{'Model err':>20}{cmp_label+' err':>18}"
                  f"{'Reduction':>14}")
        print(header)
        print("   " + "─" * (len(header) - 3))

        tot_ft = 0
        tot_cmp = 0
        tot_n = 0
        per_cat = {}

        for cat in ['N', 'A']:
            mask = (
                (m_data['CATEGORY'] == cat) & (p_data['CATEGORY'] == cat) &
                m_data['evaluation_result'].isin(['Reasonable', 'Unreasonable']) &
                p_data['evaluation_result'].isin(['Reasonable', 'Unreasonable'])
            )
            sub_m = m_data[mask]
            sub_p = p_data[mask]
            n_cat = len(sub_m)
            err_m  = int((sub_m['evaluation_result']  == 'Unreasonable').sum())
            err_cmp = int((sub_p['evaluation_result'] == 'Unreasonable').sum())
            red = ((err_cmp - err_m) / err_cmp * 100) if err_cmp > 0 else float('nan')

            red_str = f"{red:>12.1f}%" if not np.isnan(red) else f"{'N/A':>13}"
            print(f"   {cat:<12}{n_cat:>10}{err_m:>20}{err_cmp:>18}{red_str:>14}")

            per_cat[cat] = {
                'n_paired': n_cat,
                'model_errors': err_m,
                f'{cmp_label.lower()}_errors': err_cmp,
                'reduction_%': round(red, 2) if not np.isnan(red) else None,
            }
            tot_ft += err_m
            tot_cmp += err_cmp
            tot_n += n_cat

        red_tot = ((tot_cmp - tot_ft) / tot_cmp * 100) if tot_cmp > 0 else float('nan')
        red_tot_str = f"{red_tot:>12.1f}%" if not np.isnan(red_tot) else "  N/A"
        print("   " + "─" * (len(header) - 3))
        print(f"   {'Total N+A':<12}{tot_n:>10}{tot_ft:>20}{tot_cmp:>18}{red_tot_str:>14}")

        if tot_n > 0:
            print(f"\n   Error rate — Model: {tot_ft}/{tot_n} = {tot_ft/tot_n*100:.2f}%")
            print(f"   Error rate — {cmp_label}:   {tot_cmp}/{tot_n} = {tot_cmp/tot_n*100:.2f}%")
        print("═" * 80)

        self.error_analysis_results = {
            'comparator': cmp_label,
            'total_n_paired': tot_n,
            'total_model_errors': tot_ft,
            f'total_{cmp_label.lower()}_errors': tot_cmp,
            'total_reduction_%': round(red_tot, 2) if not np.isnan(red_tot) else None,
            'per_category': per_cat,
        }
        return self.error_analysis_results

    # ───────── Text report ─────────
    def generate_comparison_report(self):
        ft = self.model_metrics
        bl = self.physician_metrics
        if ft is None or bl is None:
            print("❌ Run evaluation first")
            return

        print("\n" + "═" * 80)
        print("📊 COMPARATIVE REPORT: Model vs Physician")
        print("═" * 80)

        # Header
        print(f"\n   Total cases — Model: {ft['total_cases']}  |  Physician: {bl['total_cases']}")

        # ───── Part 1: Per-category appropriateness ─────
        print("\n" + "─" * 80)
        print("🎯 Part 1: Per-category Appropriateness Rate (A and N)")
        print("─" * 80)
        print(f"\n   {'Category':<12} {'Model':>18} {'Physician':>18} {'Δ (FT−BL)':>18}")
        print(f"   {'─'*66}")
        for cat in ['N', 'A']:
            m_rate = ft.get(f'{cat}_appropriateness_rate', np.nan)
            p_rate = bl.get(f'{cat}_appropriateness_rate', np.nan)
            ft_n = ft.get(f'{cat}_total', 0)
            bl_n = bl.get(f'{cat}_total', 0)
            delta = m_rate - p_rate if not (np.isnan(m_rate) or np.isnan(p_rate)) else np.nan
            m_str = f"{m_rate:.1f}% (n={ft_n})" if not np.isnan(m_rate) else "N/A"
            p_str = f"{p_rate:.1f}% (n={bl_n})" if not np.isnan(p_rate) else "N/A"
            d_str = f"{delta:+.1f} pp" if not np.isnan(delta) else "N/A"
            print(f"   {cat:<12} {m_str:>18} {p_str:>18} {d_str:>18}")

        # ───── Part 2: S category use rate ─────
        print("\n" + "─" * 80)
        print("💊 Part 2: S Category Antibiotic Use Rate (gray zone)")
        print("─" * 80)
        m_s, p_s = ft['S_use_rate'], bl['S_use_rate']
        delta_s = m_s - p_s if not (np.isnan(m_s) or np.isnan(p_s)) else np.nan
        print(f"\n   Model use rate: {m_s:.2f}%  ({ft['S_used']}/{ft['S_total']})")
        print(f"   Physician   use rate: {p_s:.2f}%  ({bl['S_used']}/{bl['S_total']})")
        print(f"   Difference (Model − Physician): {delta_s:+.2f} pp")
        if not np.isnan(delta_s):
            if delta_s > 0:
                print(f"   → Model model is MORE likely to prescribe antibiotics in S category")
            elif delta_s < 0:
                print(f"   → Physician model is MORE likely to prescribe antibiotics in S category")
            else:
                print(f"   → Model and Physician show equal tendency")

        # ───── Part 3: Overall rates (full-dataset denominator) ─────
        print("\n" + "─" * 80)
        print(f"🎯 Part 3: Overall Rates (denominator = full dataset)")
        print("─" * 80)
        rows = [
            ('Appropriateness rate',   'overall_appropriateness_rate'),
            ('Inappropriate rate',     'overall_inappropriate_rate'),
            ('S proportion',           'overall_s_proportion'),
        ]
        print(f"\n   {'Metric':<25} {'Model':>14} {'Physician':>14} {'Δ':>12}")
        print(f"   {'─'*65}")
        for label, key in rows:
            ftv, blv = ft[key], bl[key]
            delta = ftv - blv
            print(f"   {label:<25} {ftv:>13.2f}% {blv:>13.2f}% {delta:>+11.2f} pp")

        # ───── Part 4: Overall antibiotic use tendency ─────
        print("\n" + "─" * 80)
        print("📈 Part 4: Overall Antibiotic Use Tendency")
        print("─" * 80)
        print(f"\n   Model: Yes={ft['yes_count']}  No={ft['no_count']}  "
              f"→ Overall use rate: {ft['overall_use_rate']:.1f}%")
        print(f"   Physician:   Yes={bl['yes_count']}  No={bl['no_count']}  "
              f"→ Overall use rate: {bl['overall_use_rate']:.1f}%")

    # ───────── Excel export ─────────
    def save_comparison_excel(self, output_path="model_vs_physician_comparison.xlsx"):
        if self.model_metrics is None:
            print("❌ Run evaluation first")
            return
        ft, bl = self.model_metrics, self.physician_metrics

        # Build a single tidy comparison table
        rows = []

        # Per-category appropriateness
        for cat in ['N', 'A']:
            rows.append({
                'Metric': f'{cat} appropriateness rate',
                'Category': cat,
                'Model': f"{ft[f'{cat}_appropriateness_rate']:.2f}%  ({ft[f'{cat}_reasonable']}/{ft[f'{cat}_total']})",
                'Physician':   f"{bl[f'{cat}_appropriateness_rate']:.2f}%  ({bl[f'{cat}_reasonable']}/{bl[f'{cat}_total']})",
                'Δ (Model - Physician, pp)': round(ft[f'{cat}_appropriateness_rate'] - bl[f'{cat}_appropriateness_rate'], 2),
                'Model_rate_num': round(ft[f'{cat}_appropriateness_rate'], 2),
                'Physician_rate_num':  round(bl[f'{cat}_appropriateness_rate'], 2),
            })

        # S use rate
        rows.append({
            'Metric': 'S antibiotic use rate',
            'Category': 'S',
            'Model': f"{ft['S_use_rate']:.2f}%  ({ft['S_used']}/{ft['S_total']})",
            'Physician':   f"{bl['S_use_rate']:.2f}%  ({bl['S_used']}/{bl['S_total']})",
            'Δ (Model - Physician, pp)': round(ft['S_use_rate'] - bl['S_use_rate'], 2),
            'Model_rate_num': round(ft['S_use_rate'], 2),
            'Physician_rate_num':  round(bl['S_use_rate'], 2),
        })

        # Overall (full dataset denominator)
        for label, key in [('Overall appropriateness rate', 'overall_appropriateness_rate'),
                           ('Overall inappropriate rate',   'overall_inappropriate_rate'),
                           ('Overall S proportion',         'overall_s_proportion'),
                           ('Overall antibiotic use rate',  'overall_use_rate')]:
            rows.append({
                'Metric': label,
                'Category': 'All (full dataset)',
                'Model': f"{ft[key]:.2f}%",
                'Physician':   f"{bl[key]:.2f}%",
                'Δ (Model - Physician, pp)': round(ft[key] - bl[key], 2),
                'Model_rate_num': round(ft[key], 2),
                'Physician_rate_num':  round(bl[key], 2),
            })

        comp_df = pd.DataFrame(rows)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            comp_df.to_excel(writer, sheet_name='Comparison', index=False)

            # ─── Paired test results sheet ───
            if hasattr(self, 'paired_results') and self.paired_results is not None:
                test_rows = []
                r = self.paired_results
                for test_name, display_name in [
                    ('N_appropriateness',    'N appropriateness (correct vs incorrect)'),
                    ('A_appropriateness',    'A appropriateness (correct vs incorrect)'),
                    ('S_use_rate',           'S antibiotic use rate (Yes vs No)'),
                    ('overall_use_decision', 'Overall use decision (Yes vs No)'),
                ]:
                    t = r['tests'].get(test_name, {})
                    if t.get('n', 0) == 0:
                        test_rows.append({
                            'Test': display_name, 'n_paired': 0,
                            'Note': t.get('note', 'insufficient data')
                        })
                        continue

                    row = {
                        'Test': display_name,
                        'n_paired': t['n'],
                        'Model_rate (%)': round(t.get('model_rate', t.get('model_use_rate', np.nan)), 2),
                        'Physician_rate (%)':  round(t.get('physician_rate',  t.get('physician_use_rate',  np.nan)), 2),
                        'Δ (pp)': round(t.get('delta',
                                              t.get('model_use_rate', 0) - t.get('physician_use_rate', 0)), 2),
                        'Discordant pairs (b+c)': t['discordant'],
                        'Test type': t['test_type'],
                        'Statistic': round(t['statistic'], 4),
                        'p-value': round(t['pvalue'], 4),
                        'Significance': ('***' if t['pvalue'] < 0.001
                                         else '**' if t['pvalue'] < 0.01
                                         else '*' if t['pvalue'] < 0.05 else 'ns'),
                    }
                    if 'kappa' in t:
                        row["Cohen's κ"] = round(t['kappa'], 3)
                    test_rows.append(row)

                pd.DataFrame(test_rows).to_excel(writer, sheet_name='Paired_tests', index=False)

            # Also include each source's full detailed output
            self.model_eval.evaluation_results[
                ['extracted_icd10', 'CATEGORY', 'antibiotic_used',
                 'evaluation_result', 'evaluation_reason']
            ].to_excel(writer, sheet_name='Model_detailed', index=False)
            self.physician_eval.evaluation_results[
                ['extracted_icd10', 'CATEGORY', 'antibiotic_used',
                 'evaluation_result', 'evaluation_reason']
            ].to_excel(writer, sheet_name='Physician_detailed', index=False)

        print(f"✅ Comparison Excel saved to: {output_path}")

    # ───────── Comparison figure (two panels: N/A appropriateness + S use rate) ─────────
    def plot_comparison(self, output_dir="charts"):
        if self.model_metrics is None:
            print("❌ Run evaluation first")
            return
        os.makedirs(output_dir, exist_ok=True)
        ft, bl = self.model_metrics, self.physician_metrics

        # Ensure paired tests are run
        if not hasattr(self, 'paired_results') or self.paired_results is None:
            self.run_paired_tests()
        paired = self.paired_results['tests'] if self.paired_results else {}

        def _sig_label(test_key):
            t = paired.get(test_key, {})
            if t.get('n', 0) == 0 or 'pvalue' not in t:
                return ""
            pv = t['pvalue']
            if pv < 0.001: stars = "***"
            elif pv < 0.01: stars = "**"
            elif pv < 0.05: stars = "*"
            else: stars = "ns"
            return f"p = {pv:.3f} ({stars})"

        def _draw_significance_bracket(ax, x1, x2, y, label):
            """Draw a significance bracket spanning x1→x2 at height y with label above."""
            if not label: return
            bracket_height = y * 0.04
            ax.plot([x1, x1, x2, x2],
                    [y, y + bracket_height, y + bracket_height, y],
                    lw=1.3, c='black')
            ax.text((x1 + x2) / 2, y + bracket_height * 1.3, label,
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # ─── 1×2 horizontal layout ───
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        fig.patch.set_facecolor('white')
        ax1, ax2 = axes[0], axes[1]

        # ═════════════════════════════════════════════════
        # Panel 1: Per-category appropriateness (A, N) grouped bars
        # ═════════════════════════════════════════════════
        cats = ['N', 'A']
        cat_labels = [f"{c}\n({self.model_eval._get_category_meaning(c).split()[0]})"
                      for c in cats]
        m_rates = [ft[f'{c}_appropriateness_rate'] for c in cats]
        p_rates = [bl[f'{c}_appropriateness_rate'] for c in cats]
        m_ns    = [ft[f'{c}_total'] for c in cats]
        p_ns    = [bl[f'{c}_total'] for c in cats]

        x = np.arange(len(cats))
        width = 0.35
        b1 = ax1.bar(x - width/2, m_rates, width,
                     label=f'Model (n per cat: {m_ns})',
                     color=COMPARISON_COLORS['Model'], alpha=0.88,
                     edgecolor='white', linewidth=2)
        b2 = ax1.bar(x + width/2, p_rates, width,
                     label=f'Physician (n per cat: {p_ns})',
                     color=COMPARISON_COLORS['Physician'], alpha=0.88,
                     edgecolor='white', linewidth=2)

        ax1.set_title('A — Appropriateness Rate by BMJ Category\n'
                      'Paired McNemar test',
                      fontsize=13, fontweight='bold', pad=15)
        ax1.set_ylabel('Appropriateness Rate (%)', fontsize=11)
        ax1.set_xticks(x); ax1.set_xticklabels(cat_labels, fontsize=11)
        ax1.set_ylim(0, 120)
        ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax1.text(bar.get_x() + bar.get_width()/2., h + 2,
                             f'{h:.1f}%', ha='center', va='bottom',
                             fontsize=10, fontweight='bold')
        for i, cat in enumerate(cats):
            label = _sig_label(f'{cat}_appropriateness')
            y_pos = max(m_rates[i] if not np.isnan(m_rates[i]) else 0,
                        p_rates[i] if not np.isnan(p_rates[i]) else 0) + 8
            _draw_significance_bracket(ax1, x[i] - width/2, x[i] + width/2, y_pos, label)

        # ═════════════════════════════════════════════════
        # Panel 2: S category antibiotic use rate (the gray zone)
        # ═════════════════════════════════════════════════
        s_values = [ft['S_use_rate'], bl['S_use_rate']]
        s_labels = [f"Model\n(n={ft['S_total']})", f"Physician\n(n={bl['S_total']})"]
        s_colors = [COMPARISON_COLORS['Model'], COMPARISON_COLORS['Physician']]
        bars = ax2.bar(s_labels, s_values, color=s_colors, alpha=0.88,
                       edgecolor='white', linewidth=2, width=0.55)
        ax2.set_title('B — Antibiotic Use Rate in S Category (gray zone)\n'
                      'Paired McNemar test',
                      fontsize=13, fontweight='bold', pad=15)
        ax2.set_ylabel('Antibiotic Use Rate (%)', fontsize=11)
        ax2.set_ylim(0, 120)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, v in zip(bars, s_values):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width()/2., v + 2,
                         f'{v:.1f}%', ha='center', va='bottom',
                         fontsize=11, fontweight='bold')
        s_sig_label = _sig_label('S_use_rate')
        if s_sig_label:
            max_s = max(v for v in s_values if not np.isnan(v))
            _draw_significance_bracket(ax2, 0, 1, max_s + 8, s_sig_label)

        delta = ft['S_use_rate'] - bl['S_use_rate']
        if not np.isnan(delta):
            ax2.text(0.02, 0.98, f'Δ = {delta:+.1f} pp',
                     transform=ax2.transAxes, ha='left', va='top',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0',
                               edgecolor='#FB8C00', linewidth=1.5))

        # Common styling
        for ax in axes:
            ax.set_facecolor(BLUE_COLORS['backgrounds'])
            for s in ['top', 'right']: ax.spines[s].set_visible(False)
            for s in ['left', 'bottom']:
                ax.spines[s].set_color('#DDDDDD')
                ax.spines[s].set_linewidth(1)

        # Legend for colors (shared)
        legend_elements = [
            Patch(facecolor=COMPARISON_COLORS['Model'], label='Model'),
            Patch(facecolor=COMPARISON_COLORS['Physician'],  label='Physician'),
        ]

        plt.tight_layout(pad=2.5, w_pad=3.0)
        plt.subplots_adjust(top=0.85, bottom=0.12)
        fig.suptitle('Model vs Physician — Antibiotic Recommendation (BMJ-based)',
                     fontsize=15, fontweight='bold', y=0.98)

        out_path = os.path.join(output_dir, "model_vs_physician_comparison.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Comparison figure saved to: {out_path}")
        plt.show()

    # ───────── Helper: decision agreement matrix (retained for Excel/backend use) ─────────
    def _compute_agreement_matrix(self):
        """
        Align model and physician cases row by row (both files must be the same cases,
        in the same order). If lengths mismatch, we align by index position only as far
        as the shorter length.
        Returns a 2x2 agreement matrix on Yes/No decisions,
        plus agreement rate and Cohen's kappa.
        """
        m_df = self.model_eval.evaluation_results
        p_df = self.physician_eval.evaluation_results
        if m_df is None or p_df is None:
            return None

        # Align by row index — assumes both files are the same cases in same order
        n = min(len(m_df), len(p_df))
        m_use = m_df['antibiotic_used'].iloc[:n].values
        p_use = p_df['antibiotic_used'].iloc[:n].values

        # Keep only rows where both sides have Yes/No
        mask = np.array([(fu in ('Yes', 'No')) and (bu in ('Yes', 'No'))
                         for fu, bu in zip(m_use, p_use)])
        if mask.sum() == 0:
            return None
        m_use = m_use[mask]; p_use = p_use[mask]

        # Build 2x2 matrix: rows = Model No/Yes, cols = Physician No/Yes
        mat = np.zeros((2, 2), dtype=int)
        for fu, bu in zip(m_use, p_use):
            i = 1 if fu == 'Yes' else 0
            j = 1 if bu == 'Yes' else 0
            mat[i, j] += 1

        total = mat.sum()
        agree = mat[0, 0] + mat[1, 1]
        agreement_rate = agree / total * 100 if total else 0

        # Cohen's kappa
        po = agree / total if total else 0
        p_m_yes = (mat[1, 0] + mat[1, 1]) / total if total else 0
        p_p_yes = (mat[0, 1] + mat[1, 1]) / total if total else 0
        pe = (p_m_yes * p_p_yes) + ((1 - p_m_yes) * (1 - p_p_yes))
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan

        note = ("Cases aligned by row index; both files must contain the same cases "
                "in the same order for correct matching.")

        return {
            'matrix': mat,
            'total': total,
            'agreement_rate': agreement_rate,
            'kappa': kappa,
            'note': note,
        }


# ═════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════
def main():
    print("🏥 Antibiotic Recommendation — Model vs Physician Comparison (BMJ)")
    print("=" * 70)

    # ─── File paths (modify as needed) ────────────────────────
    standard_file   = r"C:\Users\38674\Desktop\23-24\23-24balanced-eva\bmj抗生素使用编码.xlsx"
    model_file  = r"C:\Users\38674\Desktop\23-24\23-24balanced-eva\大模型评估模型结果output_with_detailed_scores.xlsx"  # ← model output
    physician_file   = r"C:\Users\38674\Desktop\23-24\23-24balanced-eva\大模型评估医生output_with_detailed_scores.xlsx"       # ← physician (EHR) output

    output_excel    = "model_vs_physician_comparison.xlsx"
    output_dir      = r"C:\Users\38674\Desktop\charts"

    comparator = ComparativeEvaluator()
    if not comparator.run(standard_file, model_file, physician_file):
        return

    # Descriptive comparison report
    comparator.generate_comparison_report()

    # ⭐ Paired statistical tests (McNemar)
    comparator.run_paired_tests()
    comparator.print_paired_tests()

    # ⭐ Error analysis — absolute error counts and reduction rate
    comparator.run_error_analysis()

    # Excel output (includes Paired_tests sheet)
    comparator.save_comparison_excel(output_excel)

    # Comparison figure with significance annotations
    comparator.plot_comparison(output_dir=output_dir)

    print("\n" + "═" * 70)
    print("✅ All done.")
    print("═" * 70)


if __name__ == "__main__":
    main()