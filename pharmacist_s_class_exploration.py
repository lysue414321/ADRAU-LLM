import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和蓝色系配色
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BLUE_COLORS = {
    'primary': ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0',
                '#0D47A1'],
    'chart_colors': {
        '合理': '#2196F3',
        '不合理': '#1565C0',
        '部分合理': '#64B5F6',
        '完全合理': '#1976D2',
        '一致': '#4CAF50',
        '不一致': '#F44336',
        '可接受': '#FF9800',
        '是': '#1976D2',
        '否': '#90CAF9',
        '未知': '#B0BEC5',
        '完全一致': '#4CAF50',
        '基本合理': '#FF9800',
        '存在偏差': '#F44336',
        '无法评估': '#B0BEC5',
        '无法比较': '#78909C'
    },
    'category_colors': ['#E3F2FD', '#BBDEFB', '#64B5F6', '#1976D2', '#1565C0', '#0D47A1'],
    'backgrounds': '#FAFAFA'
}


class AntibioticCrossChecker:
    def __init__(self):
        self.bmj_standard_df = None
        self.clinical_data_df = None
        self.cross_check_results = None
        self.s_rationality_results = None   # ★ 新增：保存 S 类合理性分析结果

    def load_bmj_standard(self, file_path):
        """加载BMJ标准文件"""
        try:
            self.bmj_standard_df = pd.read_excel(file_path)
            print(f"✅ 成功加载BMJ标准文件，共 {len(self.bmj_standard_df)} 条记录")
            print(f"📊 BMJ抗生素使用标准分布：")
            category_counts = self.bmj_standard_df['CATEGORY'].value_counts()
            for category, count in category_counts.items():
                meaning = self._get_category_meaning(category)
                print(f"   {category} ({meaning}): {count} 条")
            return True
        except Exception as e:
            print(f"❌ 加载BMJ标准文件失败: {e}")
            return False

    def load_clinical_data(self, file_path):
        """加载临床数据文件"""
        try:
            self.clinical_data_df = pd.read_excel(file_path)
            print(f"✅ 成功加载临床数据文件，共 {len(self.clinical_data_df)} 条记录")

            # 显示列名
            print(f"📋 数据文件包含的列：")
            for i, col in enumerate(self.clinical_data_df.columns):
                print(f"   {i + 1}. {col}")

            required_columns = ['ICD10编码', '是否使用抗生素', '评分1']
            missing_columns = [col for col in required_columns if col not in self.clinical_data_df.columns]
            if missing_columns:
                print(f"⚠️ 缺少必需的列: {missing_columns}")
                return False

            self._preprocess_clinical_data()
            self._show_data_distribution()
            return True
        except Exception as e:
            print(f"❌ 加载临床数据文件失败: {e}")
            return False

    def _preprocess_clinical_data(self):
        """预处理临床数据"""
        self.clinical_data_df['cleaned_icd10'] = self.clinical_data_df['ICD10编码'].apply(
            self._extract_icd10_code
        )
        self.clinical_data_df['antibiotic_used_std'] = self.clinical_data_df['是否使用抗生素'].apply(
            self._standardize_antibiotic_field
        )
        self.clinical_data_df['score1_std'] = self.clinical_data_df['评分1'].apply(
            self._standardize_score_field
        )
        if '评分2' in self.clinical_data_df.columns:
            self.clinical_data_df['score2_std'] = self.clinical_data_df['评分2'].apply(
                self._standardize_score_field
            )
        else:
            self.clinical_data_df['score2_std'] = '未知'

    def _extract_icd10_code(self, code_text):
        """从文本中提取ICD10编码"""
        if pd.isna(code_text):
            return None
        code_str = str(code_text).strip()
        patterns = [
            r'([A-Z]\d{2}\.?\d*)',
            r'疾病编码[:：]\s*([A-Z]\d{2}\.?\d*)',
            r'ICD[:：]?\s*([A-Z]\d{2}\.?\d*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, code_str, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return code_str.upper().strip()

    def _standardize_antibiotic_field(self, value):
        """标准化抗生素使用字段"""
        if pd.isna(value):
            return "未知"
        value_str = str(value).strip().lower()
        if value_str in ['是', 'yes', 'y', '1', 'true', '使用', '用']:
            return "是"
        elif value_str in ['否', 'no', 'n', '0', 'false', '不使用', '未使用']:
            return "否"
        else:
            return "未知"

    def _standardize_score_field(self, value):
        """标准化评分字段：0=不合理，1=部分合理，2=完全合理"""
        if pd.isna(value):
            return "未知"
        try:
            score = int(float(value))
            if score == 0:
                return "不合理"
            elif score == 1:
                return "部分合理"
            elif score == 2:
                return "完全合理"
            else:
                return "未知"
        except:
            return "未知"

    def _get_category_meaning(self, category):
        """获取BMJ类别含义"""
        meanings = {
            'N': '从不使用抗生素',
            'S': '有时使用抗生素',
            'A': '总是使用抗生素'
        }
        return meanings.get(category, '未知类别')

    def _show_data_distribution(self):
        """显示数据分布"""
        print(f"\n📊 临床数据分布：")
        antibiotic_counts = self.clinical_data_df['antibiotic_used_std'].value_counts()
        print(f"\n抗生素使用分布：")
        for usage, count in antibiotic_counts.items():
            percentage = count / len(self.clinical_data_df) * 100
            print(f"   {usage}: {count} 例 ({percentage:.1f}%)")

        score_counts = self.clinical_data_df['score1_std'].value_counts()
        print(f"\n评分1分布：")
        for score, count in score_counts.items():
            percentage = count / len(self.clinical_data_df) * 100
            print(f"   {score}: {count} 例 ({percentage:.1f}%)")

    def perform_cross_check(self):
        """执行交叉检查"""
        if self.bmj_standard_df is None or self.clinical_data_df is None:
            print("❌ 请先加载BMJ标准文件和临床数据文件")
            return False

        merged_df = self.clinical_data_df.merge(
            self.bmj_standard_df,
            left_on='cleaned_icd10',
            right_on='ICD10_CODE',
            how='left'
        )

        def cross_check_logic(row):
            bmj_category = row['CATEGORY']
            antibiotic_used = row['antibiotic_used_std']
            clinical_score = row['score1_std']

            # 基础字段缺失判断
            if pd.isna(bmj_category):
                return {
                    'bmj_recommendation': '无标准',
                    'bmj_vs_actual': '无法比较',
                    'clinical_vs_bmj': '无法比较',
                    'clinical_vs_bmj_strict': '无法比较',
                    'triple_consistency': '无法评估',
                    'check_details': 'ICD10编码未在BMJ标准中找到'
                }
            if antibiotic_used == "未知":
                return {
                    'bmj_recommendation': self._get_category_meaning(bmj_category),
                    'bmj_vs_actual': '无法比较',
                    'clinical_vs_bmj': '无法比较',
                    'clinical_vs_bmj_strict': '无法比较',
                    'triple_consistency': '无法评估',
                    'check_details': '抗生素使用情况未知'
                }
            if clinical_score == "未知":
                return {
                    'bmj_recommendation': self._get_category_meaning(bmj_category),
                    'bmj_vs_actual': '无法比较',
                    'clinical_vs_bmj': '无法比较',
                    'clinical_vs_bmj_strict': '无法比较',
                    'triple_consistency': '无法评估',
                    'check_details': '临床评分未知'
                }

            bmj_recommendation = self._get_category_meaning(bmj_category)
            bmj_vs_actual = self._compare_bmj_vs_actual(bmj_category, antibiotic_used)
            clinical_vs_bmj = self._compare_clinical_vs_bmj_original(clinical_score, bmj_category, antibiotic_used)
            clinical_vs_bmj_strict = self._compare_clinical_vs_bmj_strict(clinical_score, bmj_category, antibiotic_used)
            triple_consistency = self._evaluate_triple_consistency(bmj_category, antibiotic_used, clinical_score)
            check_details = f"BMJ:{bmj_category}, 实际使用:{antibiotic_used}, 临床评分:{clinical_score}"

            return {
                'bmj_recommendation': bmj_recommendation,
                'bmj_vs_actual': bmj_vs_actual,
                'clinical_vs_bmj': clinical_vs_bmj,
                'clinical_vs_bmj_strict': clinical_vs_bmj_strict,
                'triple_consistency': triple_consistency,
                'check_details': check_details
            }

        cross_check_results = merged_df.apply(cross_check_logic, axis=1, result_type='expand')
        self.cross_check_results = pd.concat([merged_df, cross_check_results], axis=1)

        print(f"✅ 交叉检查完成，共 {len(self.cross_check_results)} 条记录")
        self._print_summary()
        return True

    def _compare_bmj_vs_actual(self, bmj_category, antibiotic_used):
        """BMJ vs 实际使用"""
        if bmj_category == 'N':
            return '一致' if antibiotic_used == '否' else '不一致'
        elif bmj_category == 'A':
            return '一致' if antibiotic_used == '是' else '不一致'
        elif bmj_category == 'S':
            return '可接受'
        else:
            return '无法比较'

    def _compare_clinical_vs_bmj_original(self, clinical_score, bmj_category, antibiotic_used):
        """原始临床评分 vs BMJ标准"""
        if bmj_category not in ['A', 'N']:
            return '无法比较'
        if clinical_score not in ['部分合理', '完全合理']:
            return '无法比较'

        if bmj_category == 'A':
            correct_decision = (antibiotic_used == '是')
        elif bmj_category == 'N':
            correct_decision = (antibiotic_used == '否')
        else:
            return '无法比较'

        return '一致' if correct_decision else '不一致'

    def _compare_clinical_vs_bmj_strict(self, clinical_score, bmj_category, antibiotic_used):
        """严格评估模式下的临床评分 vs BMJ标准"""
        if bmj_category not in ['A', 'N']:
            return '无法比较'
        if clinical_score not in ['不合理', '完全合理']:
            return '无法比较'

        if bmj_category == 'A':
            bmj_compliant = (antibiotic_used == '是')
        elif bmj_category == 'N':
            bmj_compliant = (antibiotic_used == '否')

        if bmj_compliant and clinical_score == '完全合理':
            return '一致'
        elif not bmj_compliant and clinical_score == '不合理':
            return '一致'
        else:
            return '不一致'

    def _evaluate_triple_consistency(self, bmj_category, antibiotic_used, clinical_score):
        """三重一致性评估"""
        actual_match = self._compare_bmj_vs_actual(bmj_category, antibiotic_used)
        clinical_match = self._compare_clinical_vs_bmj_original(clinical_score, bmj_category, antibiotic_used)

        if actual_match == '一致' and clinical_match == '一致':
            return '完全一致'
        elif actual_match == '可接受' and clinical_match == '一致':
            return '基本合理'
        else:
            return '存在偏差'

    def analyze_s_category_distribution(self):
        """分析S类别（有时使用抗生素）的详细分布情况"""
        if self.cross_check_results is None:
            print("❌ 请先执行交叉检查")
            return

        s_category_data = self.cross_check_results[self.cross_check_results['CATEGORY'] == 'S']

        if len(s_category_data) == 0:
            print("⚠️ 没有找到BMJ类别为S的病例")
            return

        print(f"\n🔍 S类别（有时使用抗生素）详细分析:")
        print("=" * 80)
        print(f"S类别病例总数: {len(s_category_data)} 例")

        antibiotic_usage = s_category_data['antibiotic_used_std'].value_counts()
        print(f"\n📊 S类别中抗生素实际使用情况:")
        print("-" * 50)
        print(f"{'使用情况':<15} {'病例数':<10} {'占S类比例':<15}")
        print("-" * 50)
        for usage, count in antibiotic_usage.items():
            percentage = count / len(s_category_data) * 100
            print(f"{usage:<15} {count:<10} {percentage:>6.1f}%")
        print("-" * 50)

        print(f"\n📊 S类别中临床评分分布:")
        clinical_scores = s_category_data['score1_std'].value_counts()
        print("-" * 50)
        print(f"{'临床评分':<15} {'病例数':<10} {'占S类比例':<15}")
        print("-" * 50)
        for score, count in clinical_scores.items():
            percentage = count / len(s_category_data) * 100
            print(f"{score:<15} {count:<10} {percentage:>6.1f}%")
        print("-" * 50)

        print(f"\n📊 S类别交叉分析（抗生素使用 vs 临床评分）:")
        cross_table = pd.crosstab(s_category_data['antibiotic_used_std'],
                                  s_category_data['score1_std'],
                                  margins=True)
        print(cross_table)

    def analyze_an_category_inconsistency(self):
        """分析A和N类别中BMJ vs 实际使用的不一致情况"""
        if self.cross_check_results is None:
            print("❌ 请先执行交叉检查")
            return

        an_data = self.cross_check_results[self.cross_check_results['CATEGORY'].isin(['A', 'N'])]
        if len(an_data) == 0:
            print("⚠️ 没有找到BMJ类别为A或N的病例")
            return

        print(f"\n🔍 A和N类别不一致性详细分析:")
        print("=" * 80)
        consistency_overall = an_data['bmj_vs_actual'].value_counts()
        print(f"A+N类别总体一致性分布:")
        total_an = len(an_data)
        for consistency, count in consistency_overall.items():
            percentage = count / total_an * 100
            print(f"   {consistency:<15} {count:<10} {percentage:>6.1f}%")

        for category in ['A', 'N']:
            category_data = an_data[an_data['CATEGORY'] == category]
            if len(category_data) == 0:
                continue
            print(f"\n📊 BMJ类别-{category} ({self._get_category_meaning(category)}):")
            print(f"  总病例数: {len(category_data)}")
            consistency_dist = category_data['bmj_vs_actual'].value_counts()
            for consistency, count in consistency_dist.items():
                percentage = count / len(category_data) * 100
                print(f"    {consistency:<10} {count:<6} {percentage:>6.1f}%")

    def _print_summary(self):
        """打印摘要"""
        print(f"\n📈 交叉检查结果摘要：")
        total_cases = len(self.cross_check_results)

        category_counts = self.cross_check_results['CATEGORY'].value_counts()
        print(f"\nBMJ标准类别分布:")
        for category, count in category_counts.items():
            if pd.notna(category):
                meaning = self._get_category_meaning(category)
                percentage = count / total_cases * 100
                print(f"   {category} ({meaning}): {count} ({percentage:.1f}%)")

        no_bmj_count = self.cross_check_results['CATEGORY'].isna().sum()
        if no_bmj_count > 0:
            percentage = no_bmj_count / total_cases * 100
            print(f"   无BMJ标准: {no_bmj_count} ({percentage:.1f}%)")

        bmj_vs_actual_counts = self.cross_check_results['bmj_vs_actual'].value_counts()
        print(f"\nBMJ vs 实际使用一致性:")
        print(bmj_vs_actual_counts.to_string())

    # ═════════════════════════════════════════════════════════════════
    # ★ 新增方法 1：S 类合理性分析（微调模型决策 × 药师共识评分）
    # ═════════════════════════════════════════════════════════════════
    def analyze_s_category_rationality(self):
        """
        S 类别合理性分析 —— 仅针对微调模型的决策。
        在 BMJ=S 的病例里，用药师共识评分（score1_std）判定微调模型决策是否合理。

        输出两种判定标准下的合理率：
        - 严格标准（strict）：仅 score=2 (完全合理) 算合理
        - 宽松标准（lenient）：score>=1 (部分合理或完全合理) 算合理

        同时按"使用抗生素 / 不使用抗生素"分层展示。
        """
        if self.cross_check_results is None:
            print("❌ 请先执行 perform_cross_check()")
            return None

        # 1. 筛选 BMJ 为 S 的病例
        s_data = self.cross_check_results[self.cross_check_results['CATEGORY'] == 'S'].copy()
        if len(s_data) == 0:
            print("⚠️ 没有找到 BMJ 类别为 S 的病例")
            return None

        # 2. 剔除药师评分未知的病例
        valid_scores = ['不合理', '部分合理', '完全合理']
        s_eval = s_data[s_data['score1_std'].isin(valid_scores)].copy()
        if len(s_eval) == 0:
            print("⚠️ S 类中没有药师评分有效的病例")
            return None

        # 3. 两种标准下的合理性标记
        s_eval['strict_reasonable']  = (s_eval['score1_std'] == '完全合理')
        s_eval['lenient_reasonable'] = s_eval['score1_std'].isin(['部分合理', '完全合理'])

        total     = len(s_eval)
        strict_n  = int(s_eval['strict_reasonable'].sum())
        lenient_n = int(s_eval['lenient_reasonable'].sum())
        strict_rate  = strict_n  / total * 100
        lenient_rate = lenient_n / total * 100

        print("\n" + "=" * 80)
        print("🔬 S 类别合理性评估 —— 微调模型决策 vs 药师共识评分")
        print("=" * 80)
        print(f"   S 类病例总数:          {len(s_data)}")
        print(f"   药师评分有效病例:      {total}  (占 S 类 {total/len(s_data)*100:.1f}%)")

        # 3.1 总体合理率
        print("\n" + "-" * 80)
        print("📊 1. 总体合理率（两种判定标准）")
        print("-" * 80)
        print(f"   严格标准 (score=2 才算合理):  {strict_n}/{total} = {strict_rate:.2f}%")
        print(f"   宽松标准 (score≥1 算合理):    {lenient_n}/{total} = {lenient_rate:.2f}%")
        print(f"   Δ (宽松 − 严格):             {lenient_rate - strict_rate:+.2f} pp")

        # 3.2 按微调模型决策分层
        print("\n" + "-" * 80)
        print("📊 2. 按微调模型决策分层")
        print("-" * 80)
        print(f"   {'决策':<15}{'n':>6}{'严格合理':>20}{'宽松合理':>20}")
        layer_rows = []
        for decision_code, decision_label in [('是', '使用抗生素'), ('否', '不使用抗生素')]:
            sub = s_eval[s_eval['antibiotic_used_std'] == decision_code]
            n = len(sub)
            if n == 0:
                continue
            s_strict  = int(sub['strict_reasonable'].sum())
            s_lenient = int(sub['lenient_reasonable'].sum())
            r_strict  = s_strict  / n * 100
            r_lenient = s_lenient / n * 100
            print(f"   {decision_label:<15}{n:>6}"
                  f"{f'{s_strict}/{n} ({r_strict:.1f}%)':>20}"
                  f"{f'{s_lenient}/{n} ({r_lenient:.1f}%)':>20}")
            layer_rows.append({
                'Decision': decision_label,
                'n': n,
                'Strict_reasonable': s_strict,
                'Strict_rate_%': round(r_strict, 2),
                'Lenient_reasonable': s_lenient,
                'Lenient_rate_%': round(r_lenient, 2),
            })

        # 3.3 药师评分详细分布
        print("\n" + "-" * 80)
        print("📊 3. 药师评分详细分布（S 类可评估病例）")
        print("-" * 80)
        score_dist = s_eval['score1_std'].value_counts()
        score_order = [('不合理', 0), ('部分合理', 1), ('完全合理', 2)]
        print(f"   {'评分':<15}{'score':<8}{'n':>6}{'占比':>10}")
        for label, code in score_order:
            count = int(score_dist.get(label, 0))
            pct = count / total * 100
            print(f"   {label:<15}{code:<8}{count:>6}{f'{pct:.2f}%':>10}")

        # 3.4 保存结果
        self.s_rationality_results = {
            'total_s_cases': len(s_data),
            'evaluable': total,
            'strict_n': strict_n,
            'strict_rate': strict_rate,
            'lenient_n': lenient_n,
            'lenient_rate': lenient_rate,
            'by_decision': layer_rows,
            'score_distribution': {k: int(v) for k, v in score_dist.to_dict().items()},
            's_eval_data': s_eval,
        }
        return self.s_rationality_results

    # ═════════════════════════════════════════════════════════════════
    # ★ 新增方法 2：S 类合理性可视化
    # ═════════════════════════════════════════════════════════════════
    def plot_s_category_rationality(self, output_dir='charts'):
        """
        可视化 S 类合理性：
        Panel A — 总体合理率（严格 vs 宽松）
        Panel B — 按微调模型决策分层（使用 / 不使用 × 严格 / 宽松）
        """
        if self.s_rationality_results is None:
            print("❌ 请先运行 analyze_s_category_rationality()")
            return
        os.makedirs(output_dir, exist_ok=True)
        r = self.s_rationality_results

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')

        # ── Panel A: 总体合理率 ──
        ax1 = axes[0]
        cats  = ['Strict\n(score=2)', 'Lenient\n(score≥1)']
        rates = [r['strict_rate'], r['lenient_rate']]
        ns    = [f"{r['strict_n']}/{r['evaluable']}",
                 f"{r['lenient_n']}/{r['evaluable']}"]
        colors = ['#1565C0', '#64B5F6']
        bars = ax1.bar(cats, rates, color=colors, alpha=0.88,
                       edgecolor='white', linewidth=2, width=0.55)
        ax1.set_title(f'A — Overall Rationality of Fine-tuned Model\n'
                      f'in S Category (n = {r["evaluable"]})',
                      fontsize=13, fontweight='bold', pad=15)
        ax1.set_ylabel('Rationality Rate (%)', fontsize=11)
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, v, n_str in zip(bars, rates, ns):
            ax1.text(bar.get_x() + bar.get_width() / 2., v + 2,
                     f'{v:.1f}%\n({n_str})', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')

        # ── Panel B: 分层（使用 / 不使用） ──
        ax2 = axes[1]
        if r['by_decision']:
            decisions     = [d['Decision'] for d in r['by_decision']]
            strict_rates  = [d['Strict_rate_%']  for d in r['by_decision']]
            lenient_rates = [d['Lenient_rate_%'] for d in r['by_decision']]
            ns_list       = [d['n'] for d in r['by_decision']]

            x = np.arange(len(decisions))
            width = 0.35
            b1 = ax2.bar(x - width / 2, strict_rates, width,
                         label='Strict (score=2)', color='#1565C0',
                         alpha=0.88, edgecolor='white', linewidth=2)
            b2 = ax2.bar(x + width / 2, lenient_rates, width,
                         label='Lenient (score≥1)', color='#64B5F6',
                         alpha=0.88, edgecolor='white', linewidth=2)

            ax2.set_title('B — Rationality by Fine-tuned Model Decision\n'
                          'in S Category',
                          fontsize=13, fontweight='bold', pad=15)
            ax2.set_ylabel('Rationality Rate (%)', fontsize=11)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{d}\n(n={n})' for d, n in zip(decisions, ns_list)],
                                fontsize=10)
            ax2.set_ylim(0, 110)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            for bar, v in zip(b1, strict_rates):
                ax2.text(bar.get_x() + bar.get_width() / 2., v + 2,
                         f'{v:.1f}%', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
            for bar, v in zip(b2, lenient_rates):
                ax2.text(bar.get_x() + bar.get_width() / 2., v + 2,
                         f'{v:.1f}%', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

        # 共同样式
        for ax in axes:
            ax.set_facecolor(BLUE_COLORS['backgrounds'])
            for s in ['top', 'right']:
                ax.spines[s].set_visible(False)
            for s in ['left', 'bottom']:
                ax.spines[s].set_color('#DDDDDD')
                ax.spines[s].set_linewidth(1)

        plt.tight_layout(pad=2.5, w_pad=3.0)
        plt.subplots_adjust(top=0.85)
        fig.suptitle('S Category — Pharmacist Rationality Assessment of Fine-tuned Model',
                     fontsize=15, fontweight='bold', y=0.98)

        out_path = os.path.join(output_dir, 's_rationality_finetuned.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Figure saved to: {out_path}")
        plt.show()

    # ═════════════════════════════════════════════════════════════════
    # ★ 新增方法 3：生成 Table 3
    # ═════════════════════════════════════════════════════════════════
    def generate_table3(self):
        """
        生成 Table 3：按 BMJ 类别（N / S / A）及模型决策（Use / Non-use）
        分层统计专家评估合理率，包含：
          - Antibiotic use 列（score1_std）：严格标准 score=2；宽松标准 score≥1
          - Antibiotic choice 列（score2_std）：严格标准 score=2；宽松标准 score≥1
            * N 类（never appropriate）：use/non-use 子行均显示
            * S 类（sometimes appropriate）：仅 Use 子行显示（total 和 Non-use 行显示 /）
            * A 类（always appropriate）：use/non-use 子行均显示
        """
        if self.cross_check_results is None:
            print("❌ 请先执行 perform_cross_check()")
            return None

        df = self.cross_check_results.copy()
        valid_scores = ['不合理', '部分合理', '完全合理']

        def _stats_use(sub):
            """score1 (Antibiotic use) 严格/宽松合理率"""
            n = len(sub)
            if n == 0:
                return n, '—', '—'
            ev = sub[sub['score1_std'].isin(valid_scores)]
            n_ev = len(ev)
            if n_ev == 0:
                return n, '—', '—'
            strict_n   = int((ev['score1_std'] == '完全合理').sum())
            flexible_n = int(ev['score1_std'].isin(['部分合理', '完全合理']).sum())
            return (n_ev,
                    f"{strict_n} ({strict_n/n_ev*100:.1f})",
                    f"{flexible_n} ({flexible_n/n_ev*100:.1f})")

        def _stats_choice(sub):
            """score2 (Antibiotic choice) 严格/宽松合理率（仅 Use 病例）"""
            if 'score2_std' not in sub.columns:
                return '—', '—'
            ev = sub[sub['score2_std'].isin(valid_scores)]
            n_ev = len(ev)
            if n_ev == 0:
                return '—', '—'
            strict_n   = int((ev['score2_std'] == '完全合理').sum())
            flexible_n = int(ev['score2_std'].isin(['部分合理', '完全合理']).sum())
            return (f"{strict_n} ({strict_n/n_ev*100:.1f})",
                    f"{flexible_n} ({flexible_n/n_ev*100:.1f})")

        SLASH = '/'   # placeholder for cells where choice assessment is not applicable

        # ── 收集每一行数据 ─────────────────────────────────────────
        rows = []

        for cat, cat_label in [('N', 'Never appropriate class'),
                                ('S', 'Sometimes appropriate class'),
                                ('A', 'Always appropriate class')]:
            cat_df    = df[df['CATEGORY'] == cat]
            use_df    = cat_df[cat_df['antibiotic_used_std'] == '是']
            nonuse_df = cat_df[cat_df['antibiotic_used_std'] == '否']

            # ── 类别总行 ──────────────────────────────────────────
            n_ev_total, strict_total, flex_total = _stats_use(cat_df)

            # Antibiotic choice: S类总行不适用（显示 /），N/A 类可以计算 Use 子集
            if cat == 'S':
                choice_strict_total = SLASH
                choice_flex_total   = SLASH
            else:
                # N/A 类：总行 choice 统计仅限 Use 病例（只有使用才评估选药）
                choice_strict_total, choice_flex_total = _stats_choice(use_df)

            rows.append({
                'BMJ Category': cat_label,
                'Sub-group': '',
                'Number': n_ev_total,
                'Use_Strict':    strict_total,
                'Use_Flexible':  flex_total,
                'Choice_Strict':  choice_strict_total,
                'Choice_Flexible':choice_flex_total,
            })

            # ── Use 子行 ──────────────────────────────────────────
            n_ev_use, strict_use, flex_use = _stats_use(use_df)
            choice_strict_use, choice_flex_use = _stats_choice(use_df)

            rows.append({
                'BMJ Category': '',
                'Sub-group': '  Use',
                'Number': n_ev_use,
                'Use_Strict':    strict_use,
                'Use_Flexible':  flex_use,
                'Choice_Strict':  choice_strict_use,
                'Choice_Flexible':choice_flex_use,
            })

            # ── Non-use 子行 ──────────────────────────────────────
            n_ev_non, strict_non, flex_non = _stats_use(nonuse_df)

            # Non-use 病例不存在药物选择评估
            if cat == 'S':
                choice_strict_non = SLASH
                choice_flex_non   = SLASH
            else:
                choice_strict_non = '—'
                choice_flex_non   = '—'

            rows.append({
                'BMJ Category': '',
                'Sub-group': '  Non-use',
                'Number': n_ev_non,
                'Use_Strict':    strict_non,
                'Use_Flexible':  flex_non,
                'Choice_Strict':  choice_strict_non,
                'Choice_Flexible':choice_flex_non,
            })

        # ── 构建 DataFrame ────────────────────────────────────────
        table_df = pd.DataFrame(rows, columns=[
            'BMJ Category', 'Sub-group', 'Number',
            'Use_Strict', 'Use_Flexible',
            'Choice_Strict', 'Choice_Flexible'
        ])

        # ── 打印（宽格式）────────────────────────────────────────
        W = 130
        print("\n" + "=" * W)
        print("Table 3  Antibiotic recommendation performance against BMJ criteria and expert assessment")
        print(f"         (n={len(self.cross_check_results)})")
        print("=" * W)
        # header row 1
        print(f"{'':30}{'':>8}  {'Antibiotic use':^44}  {'Antibiotic choice':^44}")
        # header row 2
        print(f"{'BMJ Category':<30}{'Number':>8}  "
              f"{'Strict criteria':>20}  {'Flexible criteria':>20}  "
              f"{'Strict criteria':>20}  {'Flexible criteria':>20}")
        print("-" * W)
        for _, row in table_df.iterrows():
            label = row['BMJ Category'] if row['BMJ Category'] else row['Sub-group']
            print(f"{label:<30}{str(row['Number']):>8}  "
                  f"{str(row['Use_Strict']):>20}  {str(row['Use_Flexible']):>20}  "
                  f"{str(row['Choice_Strict']):>20}  {str(row['Choice_Flexible']):>20}")
        print("-" * W)
        print("Values are n (%). Strict: score = 2 (fully appropriate). Flexible: score ≥ 1.")
        print("Antibiotic choice assessed only when antibiotic was used (N/A = not applicable for non-use).")
        print("S category total and Non-use rows: choice assessment not applicable (shown as /).")
        print("=" * W)

        # ── 返回 DataFrame 供导出 ─────────────────────────────────
        self.table3_df = table_df
        return table_df

    # ═════════════════════════════════════════════════════════════════
    # 增强分析 —— 把新方法纳入流程
    # ═════════════════════════════════════════════════════════════════
    def run_enhanced_analysis(self, chart_dir='charts'):
        """运行增强分析，包括 S 类分布、A/N 不一致性、S 类合理性评估"""
        print("\n" + "=" * 80)
        print("🚀 开始增强分析")
        print("=" * 80)

        # 原有：S 类分布分析
        self.analyze_s_category_distribution()

        # 原有：A/N 类不一致性分析
        self.analyze_an_category_inconsistency()

        # ★ 新增：S 类药师合理性评估
        self.analyze_s_category_rationality()

        # ★ 新增：S 类合理性可视化
        self.plot_s_category_rationality(output_dir=chart_dir)

        # ★ 新增：生成 Table 3
        self.generate_table3()

        print("\n" + "=" * 80)
        print("✅ 增强分析完成")
        print("=" * 80)

    # ═════════════════════════════════════════════════════════════════
    # 导出结果 —— 包括新的 S 类合理性 sheet
    # ═════════════════════════════════════════════════════════════════
    def export_results(self, output_path='antibiotic_analysis_result.xlsx'):
        """导出结果到 Excel，额外导出 S 类合理性分析"""
        if self.cross_check_results is None:
            print("❌ 无结果可导出")
            return
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 原有交叉检查主结果
                self.cross_check_results.to_excel(writer, sheet_name='CrossCheck', index=False)

                # ★ S 类合理性汇总
                if self.s_rationality_results is not None:
                    r = self.s_rationality_results
                    summary_rows = [
                        {'Metric': 'S category total cases',        'Value': r['total_s_cases']},
                        {'Metric': 'Evaluable (pharmacist scored)', 'Value': r['evaluable']},
                        {'Metric': 'Strict reasonable n (score=2)', 'Value': r['strict_n']},
                        {'Metric': 'Strict rationality rate (%)',   'Value': round(r['strict_rate'], 2)},
                        {'Metric': 'Lenient reasonable n (score≥1)','Value': r['lenient_n']},
                        {'Metric': 'Lenient rationality rate (%)',  'Value': round(r['lenient_rate'], 2)},
                    ]
                    pd.DataFrame(summary_rows).to_excel(
                        writer, sheet_name='S_rationality_summary', index=False
                    )

                    if r['by_decision']:
                        pd.DataFrame(r['by_decision']).to_excel(
                            writer, sheet_name='S_rationality_by_decision', index=False
                        )

                    # 详细数据
                    r['s_eval_data'][[
                        'cleaned_icd10', 'CATEGORY', 'antibiotic_used_std',
                        'score1_std', 'strict_reasonable', 'lenient_reasonable'
                    ]].to_excel(writer, sheet_name='S_rationality_detail', index=False)

                # ★ 新增：Table 3 导出（重命名列为论文格式）
                if hasattr(self, 'table3_df') and self.table3_df is not None:
                    t3_export = self.table3_df.copy()
                    t3_export = t3_export.rename(columns={
                        'Use_Strict':     'Antibiotic use - Strict criteria',
                        'Use_Flexible':   'Antibiotic use - Flexible criteria',
                        'Choice_Strict':  'Antibiotic choice - Strict criteria',
                        'Choice_Flexible':'Antibiotic choice - Flexible criteria',
                    })
                    t3_export.to_excel(writer, sheet_name='Table3', index=False)

            print(f"✅ 结果已导出至: {output_path}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")


# ════════════════════════════════════════════════════════════
# 使用示例
# 数据文件是微调模型在 375 样本上的输出 + 药师共识评分
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    checker = AntibioticCrossChecker()

    bmj_file      = r"C:\Users\38674\Desktop\23-24\23-24balanced-eva\bmj抗生素使用编码.xlsx"
    # 这个文件里：是否使用抗生素 = 微调模型决策，评分1 = 药师共识评分
    clinical_file = r"C:\Users\38674\Desktop\正在进行\毕业设计\药师评估\抽样375.xlsx"
    chart_dir     = r"C:\Users\38674\Desktop\charts"

    if checker.load_bmj_standard(bmj_file) and checker.load_clinical_data(clinical_file):
        if checker.perform_cross_check():
            checker.run_enhanced_analysis(chart_dir=chart_dir)
            checker.export_results("antibiotic_analysis_result.xlsx")