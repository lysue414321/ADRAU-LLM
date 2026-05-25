import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings

# 设置中文字体和学术风格图表样式
import matplotlib

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
warnings.filterwarnings('ignore')


class DualDimensionPharmacistAnalyzer:
    """
    基于标准多数决策准则的双维度药师评分分析器
    分析：1. 抗生素使用决策合理性  2. 用药选择合理性

    🔍 抗生素使用判断依据：
    - 使用抗生素：处方中有2个问题（使用决策 + 用药选择）
    - 未使用抗生素：处方中只有1个问题（仅使用决策）

    加权计算说明：
    - 使用原始疾病分布（真实临床分布）作为权重
    - 反映AI推荐在真实临床场景下的合理性表现
    - 避免样本收集偏差对结果的影响
    """

    def __init__(self):
        self.data = None
        self.results = {}

        # 疾病分布（按您提供的顺序和数量）
        self.disease_distribution = [
            ('支气管炎', 86), ('变应性鼻炎', 53), ('急性上呼吸道感染', 50),
            ('急性咽炎', 45), ('急性支气管炎', 38), ('慢性鼻炎', 21),
            ('哮喘', 18), ('慢性鼻窦炎', 16), ('肺炎', 7), ('急性扁桃体炎', 7),
            ('阻塞性睡眠呼吸暂停低通气综合征', 4), ('腺样体肥大', 3),
            ('感冒（急性鼻咽炎）', 3), ('支气管扩张症', 3), ('急性喉炎', 3),
            ('慢性阻塞性肺疾病', 3), ('鼻出血', 2), ('慢性咽炎', 2),
            ('鼻中隔偏曲', 2), ('慢性扁桃体炎', 1), ('急性咽喉炎', 1),
            ('鼻和鼻窦其他特指的疾患', 1), ('上呼吸道其他疾病', 1),
            ('声带肿物、白斑', 1), ('慢性阻塞性肺疾病伴有急性加重', 1),
            ('扁桃体肥大', 1), ('慢性喉炎', 1), ('肺气肿', 1)
        ]

        # 计算原始疾病分布的总数和权重映射
        self.total_original_cases = sum(count for _, count in self.disease_distribution)
        self.original_disease_weights = {disease: count / self.total_original_cases
                                         for disease, count in self.disease_distribution}

        # 生成疾病映射表
        self.prescription_disease_mapping = self._create_prescription_mapping()

        # 两个评估维度的标识符
        self.dimension_identifiers = {
            '使用决策': "请您判断以上案例中，抗生素使用决策合理性",
            '用药选择': "请您判断以上案例中，抗生素用药选择合理性"
        }

    def _create_prescription_mapping(self):
        """创建处方编号到疾病的映射"""
        mapping = {}
        prescription_id = 1

        for disease_name, count in self.disease_distribution:
            for i in range(count):
                mapping[prescription_id] = disease_name
                prescription_id += 1

        print(f"✓ 创建疾病映射：共{len(mapping)}个处方")
        return mapping

    def load_data(self, file_path, name_col_idx=6, scenario_col_idx=17,
                  evaluation_start_idx=18, evaluation_end_idx=462):
        """加载数据并识别两个维度的处方"""
        try:
            # 读取数据
            encodings = ['utf-8', 'gbk', 'gb2312', 'cp1252']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(file_path, encoding=encoding)
                    print(f"✓ 使用 {encoding} 编码读取成功")
                    break
                except:
                    continue

            if self.data is None:
                raise ValueError("无法读取文件")

            print(f"✓ 原始数据：{len(self.data)}行 × {len(self.data.columns)}列")

            # 获取关键列
            name_col = self.data.columns[name_col_idx]
            scenario_col = self.data.columns[scenario_col_idx]

            print(f"✓ 姓名列：{name_col}")
            print(f"✓ 情景列：{scenario_col}")

            # 识别两个维度的处方结构
            prescription_structure = self._identify_prescription_structure(evaluation_start_idx, evaluation_end_idx)

            # 处理双维度评分数据
            self._process_dual_dimension_data(name_col, scenario_col, prescription_structure)

            print(f"✓ 数据处理完成")

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

        return True

    def _identify_prescription_structure(self, start_idx, end_idx):
        """识别处方结构（包含两个维度）- 改进版本"""
        prescription_structure = []
        current_prescription = {'使用决策': None, '用药选择': None, 'other_cols': []}
        prescription_count = 0

        # 遍历评估列
        for col_idx in range(start_idx, min(end_idx + 1, len(self.data.columns))):
            col_name = self.data.columns[col_idx]

            # 检查是否是使用决策列
            if self.dimension_identifiers['使用决策'] in col_name:
                # 如果之前有未完成的处方，先保存
                if current_prescription['使用决策'] is not None:
                    prescription_structure.append(current_prescription)
                    current_prescription = {'使用决策': None, '用药选择': None, 'other_cols': []}

                # 开始新处方
                prescription_count += 1
                current_prescription['使用决策'] = col_name

            # 检查是否是用药选择列
            elif any(keyword in col_name.lower() for keyword in ['用药选择', '药物选择', '选择合理']):
                current_prescription['用药选择'] = col_name

            # 其他相关列
            else:
                if current_prescription['使用决策'] is not None:  # 确保在处方范围内
                    current_prescription['other_cols'].append(col_name)

        # 保存最后一个处方
        if current_prescription['使用决策'] is not None:
            prescription_structure.append(current_prescription)

        print(f"✓ 识别到{len(prescription_structure)}个处方结构")

        # 统计结构类型
        both_questions = sum(1 for s in prescription_structure if s['用药选择'] is not None)
        decision_only = sum(1 for s in prescription_structure if s['用药选择'] is None)

        print(f"✓ 包含两个问题的处方：{both_questions}个")
        print(f"✓ 只有使用决策的处方：{decision_only}个")

        # 显示结构示例
        print(f"✓ 处方结构示例：")
        for i, structure in enumerate(prescription_structure[:3]):
            has_selection = structure['用药选择'] is not None
            print(f"  处方{i + 1}: 使用决策✓ | 用药选择{'✓' if has_selection else '✗'}")

        # 保存结构信息供后续使用
        self.prescription_structures = prescription_structure
        return prescription_structure

    def _process_dual_dimension_data(self, name_col, scenario_col, prescription_structure):
        """处理双维度评分数据"""
        self.processed_prescriptions = {
            '使用决策': [],
            '用药选择': []
        }

        # 按情景分组
        scenarios = sorted(self.data[scenario_col].unique())

        for scenario in scenarios:
            scenario_data = self.data[self.data[scenario_col] == scenario]
            pharmacists = scenario_data[name_col].tolist()

            print(f"\n🏥 情景随机组 {scenario}: {len(pharmacists)}位药师")

            # 为每个处方的每个维度收集评分
            for prescription_idx, structure in enumerate(prescription_structure):
                prescription_id = prescription_idx + 1

                # 获取疾病信息
                disease_info = self._get_prescription_disease_info(prescription_id)

                # 处理使用决策维度
                if structure['使用决策']:
                    decision_data = self._process_dimension_ratings(
                        scenario_data, name_col, structure['使用决策'],
                        prescription_id, scenario, '使用决策', disease_info
                    )
                    if decision_data:
                        self.processed_prescriptions['使用决策'].append(decision_data)

                # 处理用药选择维度
                if structure['用药选择']:
                    selection_data = self._process_dimension_ratings(
                        scenario_data, name_col, structure['用药选择'],
                        prescription_id, scenario, '用药选择', disease_info
                    )
                    if selection_data:
                        self.processed_prescriptions['用药选择'].append(selection_data)

        # 输出统计信息
        print(f"\n✓ 处理完成统计：")
        for dimension, data_list in self.processed_prescriptions.items():
            print(f"  {dimension}: {len(data_list)}个有效评估")

    def _process_dimension_ratings(self, scenario_data, name_col, rating_col,
                                   prescription_id, scenario, dimension, disease_info):
        """处理单个维度的评分"""
        ratings = []
        valid_pharmacists = []

        for _, row in scenario_data.iterrows():
            pharmacist_name = row[name_col]
            rating = self._standardize_rating(row[rating_col])

            if rating is not None:
                ratings.append(rating)
                valid_pharmacists.append(pharmacist_name)

        # 如果有足够的有效评分（至少3个）
        if len(ratings) >= 3:
            final_score, decision_type = self._apply_majority_decision_with_type(ratings)

            return {
                'prescription_id': prescription_id,
                'scenario': scenario,
                'dimension': dimension,
                'pharmacists': valid_pharmacists,
                'original_ratings': ratings,
                'final_score': final_score,
                'n_raters': len(ratings),
                'decision_type': decision_type,
                **disease_info
            }

        return None

    def _standardize_rating(self, rating):
        """标准化评分：不合理=0，部分合理=1，完全合理=2"""
        if pd.isna(rating) or rating == '' or str(rating).strip() in ['(跳过)', '跳过', '（跳过）']:
            return None

        rating_str = str(rating).strip()

        # 处理文本评分
        if '完全合理' in rating_str:
            return 2
        elif '部分合理' in rating_str:
            return 1
        elif '不合理' in rating_str:
            return 0
        else:
            # 处理数值评分
            try:
                num_val = float(rating_str)
                if num_val == 1.0:
                    return 2  # 1.0 对应完全合理
                elif num_val == 0.5:
                    return 1  # 0.5 对应部分合理
                elif num_val == 0.0:
                    return 0  # 0.0 对应不合理
                else:
                    return None
            except:
                return None

    def _apply_majority_decision_with_type(self, ratings):
        """应用标准多数决策规则并返回决策类型"""
        n_raters = len(ratings)

        # 统计各评分的人数
        counts = [0, 0, 0]  # [不合理, 部分合理, 完全合理]
        for rating in ratings:
            counts[rating] += 1

        max_count = max(counts)
        max_indices = [i for i, count in enumerate(counts) if count == max_count]

        # 根据评价者数量和分布情况应用决策规则
        if n_raters == 3:
            return self._decide_for_3_raters(counts, max_count, max_indices)
        elif n_raters == 4:
            return self._decide_for_4_raters(counts, max_count, max_indices)
        else:
            # 其他情况的通用规则
            return self._decide_general(counts, max_count, max_indices, n_raters)

    def _decide_for_3_raters(self, counts, max_count, max_indices):
        """3个药师的决策规则"""
        if max_count == 3:
            # 3个相同：完全一致
            return max_indices[0], "完全一致"
        elif max_count == 2:
            # 2个相同：取多数
            return max_indices[0], "2人多数"
        else:
            # 3个不同：取中位数
            return 1, "分散取中位"

    def _decide_for_4_raters(self, counts, max_count, max_indices):
        """4个药师的决策规则"""
        if max_count == 4:
            # 4个相同：完全一致
            return max_indices[0], "完全一致"
        elif max_count == 3:
            # 3+1分布：取多数
            return max_indices[0], "3人多数"
        elif max_count == 2:
            if len(max_indices) == 1:
                # 2+1+1分布：取多数
                return max_indices[0], "2人多数"
            else:
                # 2+2分布：保守策略（取较低评分）
                return min(max_indices), "平分保守"
        else:
            # 理论上不应该到达这里，但为安全起见
            return min(max_indices), "异常情况"

    def _decide_general(self, counts, max_count, max_indices, n_raters):
        """通用决策规则"""
        if len(max_indices) == 1:
            return max_indices[0], f"{n_raters}人中{max_count}人多数"
        else:
            # 平分情况：保守策略
            return min(max_indices), "平分保守"

    def _get_prescription_disease_info(self, prescription_id):
        """获取处方的疾病信息 - 直接使用原始疾病名称"""
        if prescription_id in self.prescription_disease_mapping:
            disease = self.prescription_disease_mapping[prescription_id]
            return {
                'disease': disease
            }
        else:
            return {
                'disease': '未知疾病'
            }

    def analyze_dimension_performance(self, dimension):
        """分析单个维度的性能"""
        data_list = self.processed_prescriptions[dimension]

        if not data_list:
            print(f"❌ {dimension}维度没有可分析的数据")
            return None

        # 提取最终评分
        final_scores = [item['final_score'] for item in data_list]
        total_cases = len(final_scores)

        # 统计各评分数量
        completely_reasonable = sum(1 for score in final_scores if score == 2)  # 完全合理
        partially_reasonable = sum(1 for score in final_scores if score == 1)  # 部分合理
        unreasonable = sum(1 for score in final_scores if score == 0)  # 不合理

        # 计算比率
        completely_reasonable_rate = completely_reasonable / total_cases * 100
        partially_reasonable_rate = partially_reasonable / total_cases * 100
        partially_above_rate = (completely_reasonable + partially_reasonable) / total_cases * 100
        unreasonable_rate = unreasonable / total_cases * 100

        # 计算平均分和标准差
        mean_score = np.mean(final_scores)
        std_score = np.std(final_scores)

        # 决策类型统计
        decision_types = [item['decision_type'] for item in data_list]
        decision_type_counts = Counter(decision_types)

        # 存储结果
        dimension_results = {
            'total_cases': total_cases,
            'mean_score': mean_score,
            'std_score': std_score,
            'completely_reasonable_count': completely_reasonable,
            'partially_reasonable_count': partially_reasonable,
            'unreasonable_count': unreasonable,
            'completely_reasonable_rate': completely_reasonable_rate,
            'partially_reasonable_rate': partially_reasonable_rate,
            'partially_above_rate': partially_above_rate,
            'unreasonable_rate': unreasonable_rate,
            'decision_type_counts': dict(decision_type_counts)
        }

        # 输出结果
        print(f"\n{'=' * 70}")
        print(f"{dimension}合理性分析")
        print(f"{'=' * 70}")
        print(f"总病例数：{total_cases:,}例")
        print(f"平均评分：{mean_score:.3f} ± {std_score:.3f}")
        print(f"完全合理：{completely_reasonable}例 ({completely_reasonable_rate:.1f}%)")
        print(f"部分合理：{partially_reasonable}例 ({partially_reasonable_rate:.1f}%)")
        print(f"不合理：{unreasonable}例 ({unreasonable_rate:.1f}%)")
        print(f"部分及以上合理率：{partially_above_rate:.1f}%")

        print(f"\n📊 决策类型分布：")
        for decision_type, count in decision_type_counts.most_common():
            print(f"  {decision_type}: {count}例 ({count / total_cases * 100:.1f}%)")

        return dimension_results

    def analyze_by_disease(self, dimension):
        """按原始疾病分析单个维度"""
        data_list = self.processed_prescriptions[dimension]
        disease_results = {}

        print(f"\n{'=' * 80}")
        print(f"{dimension} - 分疾病分析")
        print(f"{'=' * 80}")

        # 按疾病统计
        for disease in set(item['disease'] for item in data_list):
            disease_data = [item for item in data_list if item['disease'] == disease]
            final_scores = [item['final_score'] for item in disease_data]

            if len(final_scores) == 0:
                continue

            # 统计指标
            total_cases = len(final_scores)
            mean_score = np.mean(final_scores)
            std_score = np.std(final_scores)

            completely_reasonable = sum(1 for score in final_scores if score == 2)
            partially_reasonable = sum(1 for score in final_scores if score == 1)
            unreasonable = sum(1 for score in final_scores if score == 0)

            completely_reasonable_rate = completely_reasonable / total_cases * 100
            partially_above_rate = (completely_reasonable + partially_reasonable) / total_cases * 100
            unreasonable_rate = unreasonable / total_cases * 100

            disease_results[disease] = {
                'total_cases': total_cases,
                'proportion': total_cases / len(data_list) * 100,
                'mean_score': mean_score,
                'std_score': std_score,
                'completely_reasonable_rate': completely_reasonable_rate,
                'partially_above_rate': partially_above_rate,
                'unreasonable_rate': unreasonable_rate
            }

        # 输出分析表格
        self._print_disease_table(disease_results, dimension)

        return disease_results

    def _print_disease_table(self, disease_results, dimension):
        """打印疾病分析表格"""
        print(f"\n📊 {dimension} - 疾病统计表格：")
        print("-" * 140)
        print(
            f"{'疾病名称':<30} {'病例数':<8} {'比例(%)':<8} {'平均评分':<10} {'标准差':<8} {'完全合理率(%)':<14} {'部分及以上合理率(%)':<18} {'不合理率(%)':<12}")
        print("-" * 140)

        # 按病例数排序
        sorted_diseases = sorted(disease_results.items(), key=lambda x: x[1]['total_cases'], reverse=True)

        for disease, stats in sorted_diseases:
            print(f"{disease:<30} {stats['total_cases']:<8} {stats['proportion']:<8.1f} "
                  f"{stats['mean_score']:<10.3f} {stats['std_score']:<8.3f} "
                  f"{stats['completely_reasonable_rate']:<14.1f} "
                  f"{stats['partially_above_rate']:<18.1f} "
                  f"{stats['unreasonable_rate']:<12.1f}")

        print("-" * 140)

    def calculate_multiple_weighted_rates(self, dimension):
        """
        参考类别加权准确率的多种计算方法
        1. 原始分布加权（当前方法）
        2. 平衡加权（每个类别权重相等）
        3. 支持度加权（根据样本充足程度）
        4. 置信度加权（根据评估一致性）
        """
        data_list = self.processed_prescriptions[dimension]

        # 按疾病统计
        disease_data = {}
        for item in data_list:
            disease = item['disease']
            if disease not in disease_data:
                disease_data[disease] = []
            disease_data[disease].append(item['final_score'])

        print(f"\n{'=' * 90}")
        print(f"{dimension} - 多种加权方法对比分析")
        print(f"{'=' * 90}")

        results = {}

        # 1. 原始分布加权（当前方法）
        results['原始分布加权'] = self._calculate_original_weighted(disease_data)

        # 2. 平衡加权（每个疾病权重相等）
        results['平衡加权'] = self._calculate_balanced_weighted(disease_data)

        # 3. 支持度加权（根据样本充足程度）
        results['支持度加权'] = self._calculate_support_weighted(disease_data)

        # 4. 置信度加权（根据评估一致性）
        results['置信度加权'] = self._calculate_confidence_weighted(disease_data, dimension)

        # 5. 混合加权（结合原始分布和置信度）
        results['混合加权'] = self._calculate_hybrid_weighted(disease_data, dimension)

        # 输出对比表格
        self._print_weighted_comparison_table(results)

        return results

    def _calculate_original_weighted(self, disease_data):
        """原始分布加权（当前方法）"""
        weighted_completely = 0
        weighted_partially_above = 0
        total_weight = 0

        for disease, scores in disease_data.items():
            weight = self.original_disease_weights.get(disease, 0)
            if weight == 0:
                continue

            total_weight += weight
            completely_rate = sum(1 for score in scores if score == 2) / len(scores) * 100
            partially_above_rate = sum(1 for score in scores if score >= 1) / len(scores) * 100

            weighted_completely += completely_rate * weight
            weighted_partially_above += partially_above_rate * weight

        return {
            'method': '原始分布加权',
            'description': '基于真实临床疾病分布的权重',
            'weighted_completely_rate': weighted_completely,
            'weighted_partially_above_rate': weighted_partially_above,
            'coverage': total_weight * 100
        }

    def _calculate_balanced_weighted(self, disease_data):
        """平衡加权（每个疾病权重相等）- 类似Balanced Accuracy"""
        num_diseases = len(disease_data)
        equal_weight = 1.0 / num_diseases

        weighted_completely = 0
        weighted_partially_above = 0

        for disease, scores in disease_data.items():
            completely_rate = sum(1 for score in scores if score == 2) / len(scores) * 100
            partially_above_rate = sum(1 for score in scores if score >= 1) / len(scores) * 100

            weighted_completely += completely_rate * equal_weight
            weighted_partially_above += partially_above_rate * equal_weight

        return {
            'method': '平衡加权',
            'description': f'每个疾病权重相等({equal_weight:.3f})',
            'weighted_completely_rate': weighted_completely,
            'weighted_partially_above_rate': weighted_partially_above,
            'coverage': 100.0
        }

    def _calculate_support_weighted(self, disease_data):
        """支持度加权（根据样本充足程度调整权重）"""
        # 计算每个疾病的支持度得分
        support_scores = {}
        min_samples = 3  # 最少样本数
        target_samples = 20  # 目标样本数

        for disease, scores in disease_data.items():
            n_samples = len(scores)
            if n_samples < min_samples:
                support_score = 0  # 样本太少，权重为0
            elif n_samples >= target_samples:
                support_score = 1  # 样本充足，满权重
            else:
                # 线性插值
                support_score = (n_samples - min_samples) / (target_samples - min_samples)

            support_scores[disease] = support_score

        # 结合原始权重和支持度
        total_adjusted_weight = 0
        weighted_completely = 0
        weighted_partially_above = 0

        for disease, scores in disease_data.items():
            original_weight = self.original_disease_weights.get(disease, 0)
            support_score = support_scores[disease]
            adjusted_weight = original_weight * support_score
            total_adjusted_weight += adjusted_weight

            completely_rate = sum(1 for score in scores if score == 2) / len(scores) * 100
            partially_above_rate = sum(1 for score in scores if score >= 1) / len(scores) * 100

            weighted_completely += completely_rate * adjusted_weight
            weighted_partially_above += partially_above_rate * adjusted_weight

        # 归一化
        if total_adjusted_weight > 0:
            weighted_completely /= total_adjusted_weight
            weighted_partially_above /= total_adjusted_weight

        return {
            'method': '支持度加权',
            'description': f'根据样本充足程度调整(最少{min_samples}，目标{target_samples})',
            'weighted_completely_rate': weighted_completely,
            'weighted_partially_above_rate': weighted_partially_above,
            'coverage': total_adjusted_weight / sum(self.original_disease_weights.values()) * 100
        }

    def _calculate_confidence_weighted(self, disease_data, dimension):
        """置信度加权（根据评估一致性调整权重）"""
        # 获取决策类型信息
        data_list = self.processed_prescriptions[dimension]
        disease_confidence = {}

        for disease in disease_data.keys():
            disease_items = [item for item in data_list if item['disease'] == disease]
            if not disease_items:
                continue

            # 计算置信度：基于决策类型的一致性
            decision_types = [item['decision_type'] for item in disease_items]
            type_counts = Counter(decision_types)

            # 一致性越高，置信度越高
            max_consistency = max(type_counts.values())
            total_decisions = len(decision_types)
            confidence_score = max_consistency / total_decisions

            disease_confidence[disease] = confidence_score

        # 结合原始权重和置信度
        total_adjusted_weight = 0
        weighted_completely = 0
        weighted_partially_above = 0

        for disease, scores in disease_data.items():
            original_weight = self.original_disease_weights.get(disease, 0)
            confidence = disease_confidence.get(disease, 0)
            adjusted_weight = original_weight * confidence
            total_adjusted_weight += adjusted_weight

            completely_rate = sum(1 for score in scores if score == 2) / len(scores) * 100
            partially_above_rate = sum(1 for score in scores if score >= 1) / len(scores) * 100

            weighted_completely += completely_rate * adjusted_weight
            weighted_partially_above += partially_above_rate * adjusted_weight

        # 归一化
        if total_adjusted_weight > 0:
            weighted_completely /= total_adjusted_weight
            weighted_partially_above /= total_adjusted_weight

        return {
            'method': '置信度加权',
            'description': '根据评估一致性调整权重',
            'weighted_completely_rate': weighted_completely,
            'weighted_partially_above_rate': weighted_partially_above,
            'coverage': total_adjusted_weight / sum(self.original_disease_weights.values()) * 100
        }

    def _calculate_hybrid_weighted(self, disease_data, dimension):
        """混合加权（结合原始分布、支持度和置信度）"""
        # 获取支持度和置信度
        data_list = self.processed_prescriptions[dimension]

        total_adjusted_weight = 0
        weighted_completely = 0
        weighted_partially_above = 0

        for disease, scores in disease_data.items():
            # 原始权重
            original_weight = self.original_disease_weights.get(disease, 0)

            # 支持度得分
            n_samples = len(scores)
            if n_samples < 3:
                support_score = 0
            elif n_samples >= 15:
                support_score = 1
            else:
                support_score = (n_samples - 3) / (15 - 3)

            # 置信度得分
            disease_items = [item for item in data_list if item['disease'] == disease]
            decision_types = [item['decision_type'] for item in disease_items]
            type_counts = Counter(decision_types)
            confidence_score = max(type_counts.values()) / len(decision_types) if decision_types else 0

            # 混合权重：原始权重 × (支持度 + 置信度) / 2
            quality_score = (support_score + confidence_score) / 2
            adjusted_weight = original_weight * quality_score
            total_adjusted_weight += adjusted_weight

            completely_rate = sum(1 for score in scores if score == 2) / len(scores) * 100
            partially_above_rate = sum(1 for score in scores if score >= 1) / len(scores) * 100

            weighted_completely += completely_rate * adjusted_weight
            weighted_partially_above += partially_above_rate * adjusted_weight

        # 归一化
        if total_adjusted_weight > 0:
            weighted_completely /= total_adjusted_weight
            weighted_partially_above /= total_adjusted_weight

        return {
            'method': '混合加权',
            'description': '结合原始分布、支持度和置信度',
            'weighted_completely_rate': weighted_completely,
            'weighted_partially_above_rate': weighted_partially_above,
            'coverage': total_adjusted_weight / sum(self.original_disease_weights.values()) * 100
        }

    def _print_weighted_comparison_table(self, results):
        """输出加权方法对比表格"""
        print(f"\n📊 多种加权方法对比：")
        print("-" * 100)
        print(f"{'加权方法':<15} {'完全合理率':<12} {'部分及以上合理率':<18} {'覆盖度':<10} {'说明':<30}")
        print("-" * 100)

        for method_result in results.values():
            method = method_result['method']
            completely = method_result['weighted_completely_rate']
            partially_above = method_result['weighted_partially_above_rate']
            coverage = method_result['coverage']
            description = method_result['description']

            print(f"{method:<15} {completely:<11.1f}% {partially_above:<17.1f}% {coverage:<9.1f}% {description:<30}")

        print("-" * 100)

        # 方法推荐
        print(f"\n💡 方法选择建议：")
        print(f"📌 原始分布加权：反映真实临床场景，适用于总体评估")
        print(f"📌 平衡加权：每个疾病权重相等，适用于疾病间公平对比")
        print(f"📌 支持度加权：考虑样本充足性，提高结果可靠性")
        print(f"📌 置信度加权：考虑评估一致性，突出高信度结果")
        print(f"📌 混合加权：综合考虑多个因素，平衡各种需求")

    def calculate_weighted_rates(self, dimension):
        """计算某个维度的加权合理率 - 基于原始疾病分布权重"""
        data_list = self.processed_prescriptions[dimension]

        # 按疾病统计
        disease_data = {}
        for item in data_list:
            disease = item['disease']
            if disease not in disease_data:
                disease_data[disease] = []
            disease_data[disease].append(item['final_score'])

        weighted_completely_reasonable = 0
        weighted_partially_above = 0
        weighted_unreasonable = 0
        total_weight_used = 0

        print(f"\n{'=' * 80}")
        print(f"{dimension} - 加权分析结果（基于原始临床疾病分布）")
        print(f"{'=' * 80}")
        print(f"{'疾病名称':<25} {'实际病例':<8} {'原始权重':<12} {'合理率':<8} {'加权贡献':<10}")
        print("-" * 80)

        for disease, scores in disease_data.items():
            # 使用原始疾病分布的权重
            original_weight = self.original_disease_weights.get(disease, 0)

            if original_weight == 0:
                print(f"⚠️  {disease}: 未找到原始权重，跳过")
                continue

            total_weight_used += original_weight

            completely_reasonable_rate = sum(1 for score in scores if score == 2) / len(scores) * 100
            partially_above_rate = sum(1 for score in scores if score >= 1) / len(scores) * 100
            unreasonable_rate = sum(1 for score in scores if score == 0) / len(scores) * 100

            # 计算加权贡献
            weighted_completely_reasonable += completely_reasonable_rate * original_weight
            weighted_partially_above += partially_above_rate * original_weight
            weighted_unreasonable += unreasonable_rate * original_weight

            print(
                f"{disease:<25} {len(scores):<8} {original_weight * 100:<11.1f}% {partially_above_rate:<7.1f}% {partially_above_rate * original_weight:<9.1f}%")

        print("-" * 80)
        print(f"使用权重总和: {total_weight_used * 100:.1f}%")

        if total_weight_used < 0.99:  # 考虑浮点数精度
            print(f"⚠️  警告：仅使用了{total_weight_used * 100:.1f}%的权重，可能有疾病数据缺失")

        print(f"\n📊 加权结果（基于原始临床分布）：")
        print(f"加权完全合理率：{weighted_completely_reasonable:.1f}%")
        print(f"加权部分及以上合理率：{weighted_partially_above:.1f}%")
        print(f"加权不合理率：{weighted_unreasonable:.1f}%")

        # 计算实际数据分布的结果作为对比
        total_cases = len(data_list)
        actual_completely = sum(1 for item in data_list if item['final_score'] == 2) / total_cases * 100
        actual_partially_above = sum(1 for item in data_list if item['final_score'] >= 1) / total_cases * 100
        actual_unreasonable = sum(1 for item in data_list if item['final_score'] == 0) / total_cases * 100

        print(f"\n📊 对比 - 实际数据分布结果：")
        print(f"实际完全合理率：{actual_completely:.1f}%")
        print(f"实际部分及以上合理率：{actual_partially_above:.1f}%")
        print(f"实际不合理率：{actual_unreasonable:.1f}%")

        print(f"\n📈 差异分析：")
        diff_completely = weighted_completely_reasonable - actual_completely
        diff_partially = weighted_partially_above - actual_partially_above
        print(f"加权vs实际 完全合理率差异：{diff_completely:+.1f}%")
        print(f"加权vs实际 部分及以上合理率差异：{diff_partially:+.1f}%")

        # 调用多种加权方法对比
        multiple_weighted_results = self.calculate_multiple_weighted_rates(dimension)

        return {
            'weighted_completely_reasonable_rate': weighted_completely_reasonable,
            'weighted_partially_above_rate': weighted_partially_above,
            'weighted_unreasonable_rate': weighted_unreasonable,
            'actual_completely_reasonable_rate': actual_completely,
            'actual_partially_above_rate': actual_partially_above,
            'actual_unreasonable_rate': actual_unreasonable,
            'total_weight_used': total_weight_used,
            'multiple_weighted_methods': multiple_weighted_results
        }

    def create_disease_table(self, dimension):
        """创建疾病分析表格"""
        data_list = self.processed_prescriptions[dimension]
        disease_data = []

        # 按疾病统计
        for disease in set(item['disease'] for item in data_list):
            disease_items = [item for item in data_list if item['disease'] == disease]
            final_scores = [item['final_score'] for item in disease_items]

            if len(final_scores) == 0:
                continue

            total_cases = len(final_scores)
            completely_reasonable = sum(1 for score in final_scores if score == 2)
            partially_reasonable = sum(1 for score in final_scores if score == 1)
            unreasonable = sum(1 for score in final_scores if score == 0)

            disease_data.append({
                '疾病名称': disease,
                '病例数': total_cases,
                '占比(%)': f"{total_cases / len(data_list) * 100:.1f}",
                '平均评分': f"{np.mean(final_scores):.3f}",
                '标准差': f"{np.std(final_scores):.3f}",
                '完全合理率(%)': f"{completely_reasonable / total_cases * 100:.1f}",
                '部分合理率(%)': f"{partially_reasonable / total_cases * 100:.1f}",
                '不合理率(%)': f"{unreasonable / total_cases * 100:.1f}",
                '部分及以上合理率(%)': f"{(completely_reasonable + partially_reasonable) / total_cases * 100:.1f}"
            })

        # 按病例数排序
        disease_data.sort(key=lambda x: x['病例数'], reverse=True)
        df_disease = pd.DataFrame(disease_data)

        print(f"\n📋 {dimension}合理性 - 疾病分析表格")
        print("-" * 120)
        print(df_disease.to_string(index=False))
        print("-" * 120)

        return df_disease

    def classify_antibiotic_usage(self):
        """
        根据处方中问题的数量来判断是否使用抗生素：

        🔍 判断依据：
        - 使用抗生素：处方中有2个问题（使用决策 + 用药选择）
        - 未使用抗生素：处方中只有1个问题（仅使用决策，无用药选择问题）

        📝 说明：
        - 如果有用药选择问题但药师回答"跳过"，仍算作使用抗生素处方
        - 判断标准是问题的存在性，而非回答的完整性
        """
        self.antibiotic_usage_classification = {
            '使用抗生素': [],  # 包含两个维度的病历
            '未使用抗生素': []  # 只有使用决策维度的病历
        }

        # 基于已识别的处方结构确定每个处方的问题类型
        if not hasattr(self, 'prescription_structures'):
            print("⚠️  处方结构信息缺失，请先运行load_data")
            return

        print(f"\n{'=' * 80}")
        print("抗生素使用情况分类（基于处方中问题数量）")
        print(f"{'=' * 80}")
        print("🔍 分类标准：")
        print("  • 使用抗生素：处方包含2个问题（使用决策 + 用药选择）")
        print("  • 未使用抗生素：处方只有1个问题（仅使用决策）")
        print("  • 注：即使用药选择回答'跳过'，仍算使用抗生素处方")
        print("")

        # 统计处方结构
        both_questions_count = sum(1 for s in self.prescription_structures if s['用药选择'] is not None)
        decision_only_count = sum(1 for s in self.prescription_structures if s['用药选择'] is None)

        print(f"处方结构统计：")
        print(f"  包含2个问题的处方：{both_questions_count}个 → 使用抗生素")
        print(f"  只有1个问题的处方：{decision_only_count}个 → 未使用抗生素")
        print(f"  总计：{len(self.prescription_structures)}个处方")

        # 按处方结构分类，收集实际有评分数据的病历
        for decision_item in self.processed_prescriptions['使用决策']:
            prescription_id = decision_item['prescription_id']

            # 从处方结构中确定是否应该有两个问题
            if prescription_id <= len(self.prescription_structures):
                structure = self.prescription_structures[prescription_id - 1]  # 处方ID从1开始
                has_selection_question = structure['用药选择'] is not None
            else:
                print(f"⚠️  处方ID {prescription_id} 超出结构范围")
                continue

            if has_selection_question:
                # ✅ 使用抗生素：处方有2个问题（使用决策 + 用药选择）
                # 查找对应的用药选择数据（即使药师回答跳过，仍算使用抗生素）
                selection_item = next(
                    (item for item in self.processed_prescriptions['用药选择']
                     if item['prescription_id'] == prescription_id), None
                )

                self.antibiotic_usage_classification['使用抗生素'].append({
                    'prescription_id': prescription_id,
                    'disease': decision_item['disease'],
                    'scenario': decision_item['scenario'],
                    'decision_score': decision_item['final_score'],
                    'decision_type': decision_item['decision_type'],
                    'selection_score': selection_item['final_score'] if selection_item else None,
                    'selection_type': selection_item['decision_type'] if selection_item else None,
                    'n_raters': decision_item['n_raters'],
                    'has_selection_data': selection_item is not None
                })
            else:
                # ❌ 未使用抗生素：处方只有1个问题（仅使用决策，无用药选择问题）
                self.antibiotic_usage_classification['未使用抗生素'].append({
                    'prescription_id': prescription_id,
                    'disease': decision_item['disease'],
                    'scenario': decision_item['scenario'],
                    'decision_score': decision_item['final_score'],
                    'decision_type': decision_item['decision_type'],
                    'n_raters': decision_item['n_raters']
                })

        # 输出分类结果
        used_count = len(self.antibiotic_usage_classification['使用抗生素'])
        unused_count = len(self.antibiotic_usage_classification['未使用抗生素'])

        print(f"\n实际分类结果（有评分数据的病历）：")
        print(f"使用抗生素病历：{used_count}例")
        print(f"未使用抗生素病历：{unused_count}例")
        print(f"总计：{used_count + unused_count}例")

        # 检查用药选择数据覆盖率
        if used_count > 0:
            has_selection_data = sum(1 for item in self.antibiotic_usage_classification['使用抗生素']
                                     if item['has_selection_data'])
            missing_selection_data = used_count - has_selection_data

            print(f"\n用药选择数据覆盖分析：")
            print(f"  有用药选择评分：{has_selection_data}例")
            print(f"  缺少用药选择评分：{missing_selection_data}例（可能回答跳过或数据缺失）")
            print(f"  覆盖率：{has_selection_data / used_count * 100:.1f}%")

        # 检查是否符合预期
        if used_count != both_questions_count:
            missing_decision_data = both_questions_count - used_count
            print(f"\n⚠️  数据完整性检查：")
            print(f"  预期使用抗生素处方：{both_questions_count}例")
            print(f"  实际有使用决策评分：{used_count}例")
            print(f"  缺少使用决策评分：{missing_decision_data}例")

    def calculate_separated_reasonableness_rates(self):
        """
        分别计算使用抗生素和未使用抗生素的合理率

        🔍 分类依据：
        - 使用抗生素：处方中有2个问题（使用决策 + 用药选择）
        - 未使用抗生素：处方中只有1个问题（仅使用决策）

        📊 计算内容：
        - 使用抗生素：分别计算使用决策和用药选择的完全合理率和部分及以上合理率
        - 未使用抗生素：计算使用决策的完全合理率和部分及以上合理率
        """

        if not hasattr(self, 'antibiotic_usage_classification'):
            self.classify_antibiotic_usage()

        results = {}

        print(f"\n{'=' * 90}")
        print("抗生素使用情况分离计算 - 合理率详细分析")
        print(f"{'=' * 90}")

        # 1. 使用抗生素组分析
        used_data = self.antibiotic_usage_classification.get('使用抗生素', [])
        if used_data:
            print(f"\n📊 使用抗生素组合理率分析")
            print("-" * 70)
            print(f"总病例数：{len(used_data)}例")

            # 使用决策合理率计算
            decision_scores = [item['decision_score'] for item in used_data]
            decision_total = len(decision_scores)

            decision_completely = sum(1 for score in decision_scores if score == 2)
            decision_partially = sum(1 for score in decision_scores if score == 1)
            decision_unreasonable = sum(1 for score in decision_scores if score == 0)
            decision_partially_above = decision_completely + decision_partially

            decision_completely_rate = decision_completely / decision_total * 100
            decision_partially_rate = decision_partially / decision_total * 100
            decision_partially_above_rate = decision_partially_above / decision_total * 100
            decision_unreasonable_rate = decision_unreasonable / decision_total * 100

            print(f"\n🔸 使用决策合理性分析：")
            print(f"  完全合理：{decision_completely}例 ({decision_completely_rate:.1f}%)")
            print(f"  部分合理：{decision_partially}例 ({decision_partially_rate:.1f}%)")
            print(f"  不合理：{decision_unreasonable}例 ({decision_unreasonable_rate:.1f}%)")
            print(f"  部分及以上合理：{decision_partially_above}例 ({decision_partially_above_rate:.1f}%)")

            # 用药选择合理率计算
            selection_data = [item for item in used_data if item.get('selection_score') is not None]
            if selection_data:
                selection_scores = [item['selection_score'] for item in selection_data]
                selection_total = len(selection_scores)

                selection_completely = sum(1 for score in selection_scores if score == 2)
                selection_partially = sum(1 for score in selection_scores if score == 1)
                selection_unreasonable = sum(1 for score in selection_scores if score == 0)
                selection_partially_above = selection_completely + selection_partially

                selection_completely_rate = selection_completely / selection_total * 100
                selection_partially_rate = selection_partially / selection_total * 100
                selection_partially_above_rate = selection_partially_above / selection_total * 100
                selection_unreasonable_rate = selection_unreasonable / selection_total * 100

                print(f"\n🔸 用药选择合理性分析：")
                print(f"  评估病例数：{selection_total}例")
                print(f"  完全合理：{selection_completely}例 ({selection_completely_rate:.1f}%)")
                print(f"  部分合理：{selection_partially}例 ({selection_partially_rate:.1f}%)")
                print(f"  不合理：{selection_unreasonable}例 ({selection_unreasonable_rate:.1f}%)")
                print(f"  部分及以上合理：{selection_partially_above}例 ({selection_partially_above_rate:.1f}%)")

                # 数据覆盖率
                coverage_rate = selection_total / decision_total * 100
                missing_count = decision_total - selection_total
                print(f"\n📝 用药选择数据覆盖情况：")
                print(f"  覆盖率：{coverage_rate:.1f}% ({selection_total}/{decision_total})")
                if missing_count > 0:
                    print(f"  缺失数据：{missing_count}例（可能因回答跳过或数据缺失）")

                results['使用抗生素'] = {
                    'total_cases': decision_total,
                    # 使用决策
                    'decision_completely_count': decision_completely,
                    'decision_completely_rate': decision_completely_rate,
                    'decision_partially_count': decision_partially,
                    'decision_partially_rate': decision_partially_rate,
                    'decision_partially_above_count': decision_partially_above,
                    'decision_partially_above_rate': decision_partially_above_rate,
                    'decision_unreasonable_count': decision_unreasonable,
                    'decision_unreasonable_rate': decision_unreasonable_rate,
                    # 用药选择
                    'selection_total_cases': selection_total,
                    'selection_completely_count': selection_completely,
                    'selection_completely_rate': selection_completely_rate,
                    'selection_partially_count': selection_partially,
                    'selection_partially_rate': selection_partially_rate,
                    'selection_partially_above_count': selection_partially_above,
                    'selection_partially_above_rate': selection_partially_above_rate,
                    'selection_unreasonable_count': selection_unreasonable,
                    'selection_unreasonable_rate': selection_unreasonable_rate,
                    'selection_coverage_rate': coverage_rate
                }
            else:
                print(f"\n🔸 用药选择合理性分析：")
                print(f"  ⚠️ 无有效用药选择评估数据")

                results['使用抗生素'] = {
                    'total_cases': decision_total,
                    # 使用决策
                    'decision_completely_count': decision_completely,
                    'decision_completely_rate': decision_completely_rate,
                    'decision_partially_count': decision_partially,
                    'decision_partially_rate': decision_partially_rate,
                    'decision_partially_above_count': decision_partially_above,
                    'decision_partially_above_rate': decision_partially_above_rate,
                    'decision_unreasonable_count': decision_unreasonable,
                    'decision_unreasonable_rate': decision_unreasonable_rate,
                    # 用药选择（无数据）
                    'selection_total_cases': 0,
                    'selection_coverage_rate': 0
                }

        # 2. 未使用抗生素组分析
        unused_data = self.antibiotic_usage_classification.get('未使用抗生素', [])
        if unused_data:
            print(f"\n📊 未使用抗生素组合理率分析")
            print("-" * 70)
            print(f"总病例数：{len(unused_data)}例")

            # 使用决策合理率计算（未使用抗生素只有使用决策维度）
            decision_scores = [item['decision_score'] for item in unused_data]
            decision_total = len(decision_scores)

            decision_completely = sum(1 for score in decision_scores if score == 2)
            decision_partially = sum(1 for score in decision_scores if score == 1)
            decision_unreasonable = sum(1 for score in decision_scores if score == 0)
            decision_partially_above = decision_completely + decision_partially

            decision_completely_rate = decision_completely / decision_total * 100
            decision_partially_rate = decision_partially / decision_total * 100
            decision_partially_above_rate = decision_partially_above / decision_total * 100
            decision_unreasonable_rate = decision_unreasonable / decision_total * 100

            print(f"\n🔸 使用决策合理性分析：")
            print(f"  完全合理：{decision_completely}例 ({decision_completely_rate:.1f}%)")
            print(f"  部分合理：{decision_partially}例 ({decision_partially_rate:.1f}%)")
            print(f"  不合理：{decision_unreasonable}例 ({decision_unreasonable_rate:.1f}%)")
            print(f"  部分及以上合理：{decision_partially_above}例 ({decision_partially_above_rate:.1f}%)")

            print(f"\n🔸 用药选择合理性分析：")
            print(f"  N/A（未使用抗生素无用药选择评估）")

            results['未使用抗生素'] = {
                'total_cases': decision_total,
                # 使用决策
                'decision_completely_count': decision_completely,
                'decision_completely_rate': decision_completely_rate,
                'decision_partially_count': decision_partially,
                'decision_partially_rate': decision_partially_rate,
                'decision_partially_above_count': decision_partially_above,
                'decision_partially_above_rate': decision_partially_above_rate,
                'decision_unreasonable_count': decision_unreasonable,
                'decision_unreasonable_rate': decision_unreasonable_rate
            }

        # 3. 对比分析汇总表
        if '使用抗生素' in results and '未使用抗生素' in results:
            print(f"\n{'=' * 100}")
            print("📊 使用抗生素 vs 未使用抗生素 对比汇总")
            print(f"{'=' * 100}")

            # 创建对比表格
            print(f"{'指标':<35} {'使用抗生素':<20} {'未使用抗生素':<20} {'差异':<15}")
            print("-" * 100)

            # 基本信息对比
            used_total = results['使用抗生素']['total_cases']
            unused_total = results['未使用抗生素']['total_cases']
            print(f"{'总病例数':<35} {used_total:<20} {unused_total:<20} {used_total - unused_total:+d}")

            # 使用决策对比
            used_decision_completely = results['使用抗生素']['decision_completely_rate']
            unused_decision_completely = results['未使用抗生素']['decision_completely_rate']
            used_decision_partially_above = results['使用抗生素']['decision_partially_above_rate']
            unused_decision_partially_above = results['未使用抗生素']['decision_partially_above_rate']

            print(
                f"{'使用决策-完全合理率(%)':<35} {used_decision_completely:<19.1f} {unused_decision_completely:<19.1f} {used_decision_completely - unused_decision_completely:+.1f}")
            print(
                f"{'使用决策-部分及以上合理率(%)':<35} {used_decision_partially_above:<19.1f} {unused_decision_partially_above:<19.1f} {used_decision_partially_above - unused_decision_partially_above:+.1f}")

            # 用药选择单独显示（仅使用抗生素组）
            if 'selection_completely_rate' in results['使用抗生素']:
                used_selection_completely = results['使用抗生素']['selection_completely_rate']
                used_selection_partially_above = results['使用抗生素']['selection_partially_above_rate']
                selection_coverage = results['使用抗生素']['selection_coverage_rate']

                print(f"{'用药选择-完全合理率(%)':<35} {used_selection_completely:<19.1f} {'N/A':<20} {'N/A'}")
                print(
                    f"{'用药选择-部分及以上合理率(%)':<35} {used_selection_partially_above:<19.1f} {'N/A':<20} {'N/A'}")
                print(f"{'用药选择-数据覆盖率(%)':<35} {selection_coverage:<19.1f} {'N/A':<20} {'N/A'}")

            print("-" * 100)

            # 4. 关键发现和结论
            print(f"\n📈 关键发现：")

            # 使用决策对比分析
            decision_diff = used_decision_partially_above - unused_decision_partially_above
            print(f"\n🔍 使用决策合理率对比：")
            if abs(decision_diff) >= 5:  # 5%以上差异认为显著
                if decision_diff > 0:
                    print(f"  ✅ 使用抗生素组显著更优：高出{decision_diff:.1f}个百分点")
                else:
                    print(f"  ⚠️ 未使用抗生素组显著更优：高出{abs(decision_diff):.1f}个百分点")
            elif abs(decision_diff) >= 2:  # 2-5%差异认为有一定差异
                if decision_diff > 0:
                    print(f"  ↗️ 使用抗生素组略优：高出{decision_diff:.1f}个百分点")
                else:
                    print(f"  ↘️ 未使用抗生素组略优：高出{abs(decision_diff):.1f}个百分点")
            else:
                print(f"  ➡️ 两组表现基本相当：差异仅{decision_diff:+.1f}个百分点")

            # 使用抗生素组内部对比
            if 'selection_partially_above_rate' in results['使用抗生素']:
                decision_vs_selection = used_decision_partially_above - used_selection_partially_above
                print(f"\n🔍 使用抗生素组内部对比（使用决策 vs 用药选择）：")
                if abs(decision_vs_selection) >= 3:
                    if decision_vs_selection > 0:
                        print(f"  📈 使用决策表现更优：高出{decision_vs_selection:.1f}个百分点")
                    else:
                        print(f"  📉 用药选择表现更优：高出{abs(decision_vs_selection):.1f}个百分点")
                else:
                    print(f"  ⚖️ 使用决策与用药选择表现相当：差异{decision_vs_selection:+.1f}个百分点")

            # 样本构成分析
            total_all_cases = used_total + unused_total
            used_proportion = used_total / total_all_cases * 100
            print(f"\n📊 样本构成：")
            print(f"  使用抗生素：{used_total}例 ({used_proportion:.1f}%)")
            print(f"  未使用抗生素：{unused_total}例 ({100 - used_proportion:.1f}%)")
            print(f"  总计：{total_all_cases}例")

        print(f"\n{'=' * 90}")
        print("✅ 分离计算完成！")
        print(f"{'=' * 90}")

        return results

    def create_separated_rates_summary_table(self):
        """创建分离计算结果的汇总表格"""
        if not hasattr(self, 'separated_rates_results'):
            print("⚠️ 请先运行 calculate_separated_reasonableness_rates() 方法")
            return None

        results = self.separated_rates_results
        summary_data = []

        # 使用抗生素组数据
        if '使用抗生素' in results:
            used_data = results['使用抗生素']

            # 使用决策行
            summary_data.append({
                '组别': '使用抗生素',
                '评估维度': '使用决策',
                '病例数': used_data['total_cases'],
                '完全合理数': used_data['decision_completely_count'],
                '完全合理率(%)': f"{used_data['decision_completely_rate']:.1f}",
                '部分合理数': used_data['decision_partially_count'],
                '部分合理率(%)': f"{used_data['decision_partially_rate']:.1f}",
                '部分及以上合理数': used_data['decision_partially_above_count'],
                '部分及以上合理率(%)': f"{used_data['decision_partially_above_rate']:.1f}",
                '不合理数': used_data['decision_unreasonable_count'],
                '不合理率(%)': f"{used_data['decision_unreasonable_rate']:.1f}"
            })

            # 用药选择行
            if 'selection_completely_rate' in used_data:
                summary_data.append({
                    '组别': '使用抗生素',
                    '评估维度': '用药选择',
                    '病例数': used_data['selection_total_cases'],
                    '完全合理数': used_data['selection_completely_count'],
                    '完全合理率(%)': f"{used_data['selection_completely_rate']:.1f}",
                    '部分合理数': used_data['selection_partially_count'],
                    '部分合理率(%)': f"{used_data['selection_partially_rate']:.1f}",
                    '部分及以上合理数': used_data['selection_partially_above_count'],
                    '部分及以上合理率(%)': f"{used_data['selection_partially_above_rate']:.1f}",
                    '不合理数': used_data['selection_unreasonable_count'],
                    '不合理率(%)': f"{used_data['selection_unreasonable_rate']:.1f}"
                })
            else:
                summary_data.append({
                    '组别': '使用抗生素',
                    '评估维度': '用药选择',
                    '病例数': 0,
                    '完全合理数': 'N/A',
                    '完全合理率(%)': 'N/A',
                    '部分合理数': 'N/A',
                    '部分合理率(%)': 'N/A',
                    '部分及以上合理数': 'N/A',
                    '部分及以上合理率(%)': 'N/A',
                    '不合理数': 'N/A',
                    '不合理率(%)': 'N/A'
                })

        # 未使用抗生素组数据
        if '未使用抗生素' in results:
            unused_data = results['未使用抗生素']

            # 使用决策行
            summary_data.append({
                '组别': '未使用抗生素',
                '评估维度': '使用决策',
                '病例数': unused_data['total_cases'],
                '完全合理数': unused_data['decision_completely_count'],
                '完全合理率(%)': f"{unused_data['decision_completely_rate']:.1f}",
                '部分合理数': unused_data['decision_partially_count'],
                '部分合理率(%)': f"{unused_data['decision_partially_rate']:.1f}",
                '部分及以上合理数': unused_data['decision_partially_above_count'],
                '部分及以上合理率(%)': f"{unused_data['decision_partially_above_rate']:.1f}",
                '不合理数': unused_data['decision_unreasonable_count'],
                '不合理率(%)': f"{unused_data['decision_unreasonable_rate']:.1f}"
            })

            # 用药选择行（未使用抗生素组无此数据）
            summary_data.append({
                '组别': '未使用抗生素',
                '评估维度': '用药选择',
                '病例数': 'N/A',
                '完全合理数': 'N/A',
                '完全合理率(%)': 'N/A',
                '部分合理数': 'N/A',
                '部分合理率(%)': 'N/A',
                '部分及以上合理数': 'N/A',
                '部分及以上合理率(%)': 'N/A',
                '不合理数': 'N/A',
                '不合理率(%)': 'N/A'
            })

        df_separated = pd.DataFrame(summary_data)

        print(f"\n📋 分离计算结果汇总表格")
        print("-" * 150)
        print(df_separated.to_string(index=False))
        print("-" * 150)

        return df_separated

    def calculate_antibiotic_usage_rates(self):
        """分别计算使用抗生素和未使用抗生素的合理率"""

        if not hasattr(self, 'antibiotic_usage_classification'):
            self.classify_antibiotic_usage()

        results = {}

        print(f"\n{'=' * 80}")
        print("分别计算使用抗生素和未使用抗生素的合理率")
        print(f"{'=' * 80}")

        for usage_type, data_list in self.antibiotic_usage_classification.items():
            if not data_list:
                continue

            print(f"\n📊 {usage_type}合理率计算：")
            print("-" * 60)

            total_cases = len(data_list)

            # 使用决策合理率计算
            decision_scores = [item['decision_score'] for item in data_list]
            decision_completely_reasonable = sum(1 for score in decision_scores if score == 2)
            decision_partially_reasonable = sum(1 for score in decision_scores if score == 1)
            decision_unreasonable = sum(1 for score in decision_scores if score == 0)

            decision_completely_rate = decision_completely_reasonable / total_cases * 100
            decision_partially_rate = decision_partially_reasonable / total_cases * 100
            decision_partially_above_rate = (
                                                        decision_completely_reasonable + decision_partially_reasonable) / total_cases * 100
            decision_unreasonable_rate = decision_unreasonable / total_cases * 100

            print(f"总病例数：{total_cases}例")
            print(f"\n使用决策合理性：")
            print(f"  完全合理：{decision_completely_reasonable}例 ({decision_completely_rate:.1f}%)")
            print(f"  部分合理：{decision_partially_reasonable}例 ({decision_partially_rate:.1f}%)")
            print(f"  不合理：{decision_unreasonable}例 ({decision_unreasonable_rate:.1f}%)")
            print(
                f"  部分及以上合理：{decision_completely_reasonable + decision_partially_reasonable}例 ({decision_partially_above_rate:.1f}%)")

            # 保存使用决策结果
            results[usage_type] = {
                'total_cases': total_cases,
                'decision_completely_count': decision_completely_reasonable,
                'decision_partially_count': decision_partially_reasonable,
                'decision_unreasonable_count': decision_unreasonable,
                'decision_completely_rate': decision_completely_rate,
                'decision_partially_rate': decision_partially_rate,
                'decision_partially_above_rate': decision_partially_above_rate,
                'decision_unreasonable_rate': decision_unreasonable_rate
            }

            # 如果是使用抗生素，还要计算用药选择合理率
            if usage_type == '使用抗生素':
                selection_data = [item for item in data_list if item.get('selection_score') is not None]

                if selection_data:
                    selection_total = len(selection_data)
                    selection_scores = [item['selection_score'] for item in selection_data]

                    selection_completely_reasonable = sum(1 for score in selection_scores if score == 2)
                    selection_partially_reasonable = sum(1 for score in selection_scores if score == 1)
                    selection_unreasonable = sum(1 for score in selection_scores if score == 0)

                    selection_completely_rate = selection_completely_reasonable / selection_total * 100
                    selection_partially_rate = selection_partially_reasonable / selection_total * 100
                    selection_partially_above_rate = (
                                                                 selection_completely_reasonable + selection_partially_reasonable) / selection_total * 100
                    selection_unreasonable_rate = selection_unreasonable / selection_total * 100

                    print(f"\n用药选择合理性：")
                    print(f"  有效评估数：{selection_total}例")
                    print(f"  完全合理：{selection_completely_reasonable}例 ({selection_completely_rate:.1f}%)")
                    print(f"  部分合理：{selection_partially_reasonable}例 ({selection_partially_rate:.1f}%)")
                    print(f"  不合理：{selection_unreasonable}例 ({selection_unreasonable_rate:.1f}%)")
                    print(
                        f"  部分及以上合理：{selection_completely_reasonable + selection_partially_reasonable}例 ({selection_partially_above_rate:.1f}%)")

                    # 添加用药选择结果
                    results[usage_type].update({
                        'selection_total_cases': selection_total,
                        'selection_completely_count': selection_completely_reasonable,
                        'selection_partially_count': selection_partially_reasonable,
                        'selection_unreasonable_count': selection_unreasonable,
                        'selection_completely_rate': selection_completely_rate,
                        'selection_partially_rate': selection_partially_rate,
                        'selection_partially_above_rate': selection_partially_above_rate,
                        'selection_unreasonable_rate': selection_unreasonable_rate
                    })
                else:
                    print(f"\n用药选择合理性：无有效评估数据")

            print("-" * 60)

        # 对比分析
        if '使用抗生素' in results and '未使用抗生素' in results:
            print(f"\n📊 对比分析：使用抗生素 vs 未使用抗生素")
            print("=" * 80)
            print(f"{'指标':<25} {'使用抗生素':<15} {'未使用抗生素':<15} {'差异':<15}")
            print("-" * 80)

            # 病例数对比
            used_count = results['使用抗生素']['total_cases']
            unused_count = results['未使用抗生素']['total_cases']
            print(f"{'病例数':<25} {used_count:<15} {unused_count:<15} {used_count - unused_count:+d}")

            # 使用决策合理率对比
            used_decision_rate = results['使用抗生素']['decision_partially_above_rate']
            unused_decision_rate = results['未使用抗生素']['decision_partially_above_rate']
            decision_diff = used_decision_rate - unused_decision_rate
            print(
                f"{'使用决策合理率(%)':<25} {used_decision_rate:<14.1f} {unused_decision_rate:<14.1f} {decision_diff:+.1f}")

            # 完全合理率对比
            used_completely_rate = results['使用抗生素']['decision_completely_rate']
            unused_completely_rate = results['未使用抗生素']['decision_completely_rate']
            completely_diff = used_completely_rate - unused_completely_rate
            print(
                f"{'使用决策完全合理率(%)':<25} {used_completely_rate:<14.1f} {unused_completely_rate:<14.1f} {completely_diff:+.1f}")

            # 不合理率对比
            used_unreasonable_rate = results['使用抗生素']['decision_unreasonable_rate']
            unused_unreasonable_rate = results['未使用抗生素']['decision_unreasonable_rate']
            unreasonable_diff = used_unreasonable_rate - unused_unreasonable_rate
            print(
                f"{'使用决策不合理率(%)':<25} {used_unreasonable_rate:<14.1f} {unused_unreasonable_rate:<14.1f} {unreasonable_diff:+.1f}")

            print("-" * 80)

            # 结论分析
            print(f"\n📈 结论分析：")
            if decision_diff > 0:
                print(f"✓ 使用抗生素病历的决策合理率比未使用抗生素高{decision_diff:.1f}个百分点")
            elif decision_diff < 0:
                print(f"⚠ 使用抗生素病历的决策合理率比未使用抗生素低{abs(decision_diff):.1f}个百分点")
            else:
                print(f"→ 使用抗生素和未使用抗生素病历的决策合理率基本相同")

            if 'selection_partially_above_rate' in results['使用抗生素']:
                selection_rate = results['使用抗生素']['selection_partially_above_rate']
                print(f"ℹ 使用抗生素病历的用药选择合理率为{selection_rate:.1f}%")

                # 与决策合理率对比
                decision_vs_selection = used_decision_rate - selection_rate
                if decision_vs_selection > 0:
                    print(f"→ 使用决策合理率比用药选择合理率高{decision_vs_selection:.1f}个百分点")
                elif decision_vs_selection < 0:
                    print(f"→ 用药选择合理率比使用决策合理率高{abs(decision_vs_selection):.1f}个百分点")

        return results

    def compare_antibiotic_usage_vs_non_usage(self):
        """
        使用抗生素 vs 未使用抗生素的详细对比分析
        - 使用抗生素：分别计算使用决策和用药选择的完全合理率和部分及以上合理率
        - 未使用抗生素：计算使用决策的完全合理率和部分及以上合理率
        """

        if not hasattr(self, 'antibiotic_usage_classification'):
            self.classify_antibiotic_usage()

        print(f"\n{'=' * 90}")
        print("使用抗生素 vs 未使用抗生素 详细对比分析")
        print(f"{'=' * 90}")

        results = {}

        # 1. 使用抗生素组分析
        used_data = self.antibiotic_usage_classification['使用抗生素']
        if used_data:
            print(f"\n📊 使用抗生素组分析 (n={len(used_data)})")
            print("-" * 60)

            # 使用决策分析
            decision_scores = [item['decision_score'] for item in used_data]
            decision_completely = sum(1 for score in decision_scores if score == 2)
            decision_partially_above = sum(1 for score in decision_scores if score >= 1)

            decision_completely_rate = decision_completely / len(decision_scores) * 100
            decision_partially_above_rate = decision_partially_above / len(decision_scores) * 100

            print(f"使用决策合理性：")
            print(f"  完全合理：{decision_completely}/{len(decision_scores)} ({decision_completely_rate:.1f}%)")
            print(
                f"  部分及以上合理：{decision_partially_above}/{len(decision_scores)} ({decision_partially_above_rate:.1f}%)")

            # 用药选择分析（只包含有评分的病历）
            selection_data = [item for item in used_data if item['selection_score'] is not None]
            if selection_data:
                selection_scores = [item['selection_score'] for item in selection_data]
                selection_completely = sum(1 for score in selection_scores if score == 2)
                selection_partially_above = sum(1 for score in selection_scores if score >= 1)

                selection_completely_rate = selection_completely / len(selection_scores) * 100
                selection_partially_above_rate = selection_partially_above / len(selection_scores) * 100

                print(f"\n用药选择合理性 (有效评估数: {len(selection_data)})：")
                print(f"  完全合理：{selection_completely}/{len(selection_data)} ({selection_completely_rate:.1f}%)")
                print(
                    f"  部分及以上合理：{selection_partially_above}/{len(selection_data)} ({selection_partially_above_rate:.1f}%)")

                # 缺失数据提示
                missing_selection = len(used_data) - len(selection_data)
                if missing_selection > 0:
                    print(f"  注：{missing_selection}例缺少用药选择评分（可能回答跳过）")
            else:
                print(f"\n用药选择合理性：无有效评估数据")
                selection_completely_rate = None
                selection_partially_above_rate = None

            results['使用抗生素'] = {
                'total_cases': len(used_data),
                'decision_completely_rate': decision_completely_rate,
                'decision_partially_above_rate': decision_partially_above_rate,
                'selection_total_cases': len(selection_data) if selection_data else 0,
                'selection_completely_rate': selection_completely_rate,
                'selection_partially_above_rate': selection_partially_above_rate
            }

        # 2. 未使用抗生素组分析
        unused_data = self.antibiotic_usage_classification['未使用抗生素']
        if unused_data:
            print(f"\n📊 未使用抗生素组分析 (n={len(unused_data)})")
            print("-" * 60)

            # 使用决策分析
            decision_scores = [item['decision_score'] for item in unused_data]
            decision_completely = sum(1 for score in decision_scores if score == 2)
            decision_partially_above = sum(1 for score in decision_scores if score >= 1)

            decision_completely_rate = decision_completely / len(decision_scores) * 100
            decision_partially_above_rate = decision_partially_above / len(decision_scores) * 100

            print(f"使用决策合理性：")
            print(f"  完全合理：{decision_completely}/{len(decision_scores)} ({decision_completely_rate:.1f}%)")
            print(
                f"  部分及以上合理：{decision_partially_above}/{len(decision_scores)} ({decision_partially_above_rate:.1f}%)")

            results['未使用抗生素'] = {
                'total_cases': len(unused_data),
                'decision_completely_rate': decision_completely_rate,
                'decision_partially_above_rate': decision_partially_above_rate
            }

        # 3. 对比分析表格
        if '使用抗生素' in results and '未使用抗生素' in results:
            print(f"\n{'=' * 90}")
            print("📊 对比分析汇总表")
            print(f"{'=' * 90}")
            print(f"{'指标':<25} {'使用抗生素':<20} {'未使用抗生素':<20} {'差异':<15}")
            print("-" * 90)

            # 病例数对比
            used_n = results['使用抗生素']['total_cases']
            unused_n = results['未使用抗生素']['total_cases']
            print(f"{'病例数':<25} {used_n:<20} {unused_n:<20} {used_n - unused_n:+d}")

            # 使用决策对比
            used_decision_completely = results['使用抗生素']['decision_completely_rate']
            unused_decision_completely = results['未使用抗生素']['decision_completely_rate']
            used_decision_partially = results['使用抗生素']['decision_partially_above_rate']


# 完整可运行版本
# 请把您原始的类代码复制到这里，然后在最后添加以下代码：

# [您的完整 DualDimensionPharmacistAnalyzer 类代码放在这里]
# 从 import pandas as pd 开始，到类定义结束

# =================== 主函数部分（请添加到您的代码最后） ===================

def main():
    """主函数 - 请在这里修改文件路径"""

    print("🚀 药师评估分析器")
    print("=" * 50)

    # ⭐⭐⭐ 请修改这里的文件路径 ⭐⭐⭐
    file_path = r"C:\Users\38674\Desktop\药师评估\325628054_按文本_抗生素处方合理性临床药师审核_17_17.csv"

    # 修改示例（请根据您的实际情况修改上面的路径）：
    # file_path = r"C:\Users\您的用户名\Desktop\您的文件名.csv"
    # file_path = r"D:\数据\药师评估数据.csv"

    print(f"📁 数据文件: {file_path}")

    # 创建分析器并运行
    analyzer = DualDimensionPharmacistAnalyzer()

    if analyzer.load_data(file_path):
        print("✅ 数据加载成功，开始分析...")

        # 核心分析：分离计算合理率
        results = analyzer.calculate_separated_reasonableness_rates()
        analyzer.separated_rates_results = results

        # 创建汇总表
        table = analyzer.create_separated_rates_summary_table()

        # 保存到Excel
        try:
            analyzer.save_results_to_excel()
            print("✅ 结果已保存到Excel文件")
        except Exception as e:
            print(f"⚠️ Excel保存失败: {e}")

        print("\n🎉 分析完成！")

    else:
        print("❌ 数据加载失败")

    input("按回车键退出...")


# 运行程序
if __name__ == "__main__":
    main()