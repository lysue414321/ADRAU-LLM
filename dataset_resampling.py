#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗数据平衡处理脚本
保存为：medical_data_processor.py
"""

import json
import pandas as pd
import numpy as np
import random
import re
from collections import Counter, defaultdict
from sklearn.utils import resample
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalDataProcessor:
    def __init__(self, random_seed=42):
        """初始化处理器"""
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 目标样本数设置
        self.target_samples = {
            'high_freq': 8000,  # 高频疾病目标样本数
            'mid_freq': 5000,  # 中频疾病目标样本数
            'low_freq': 2500,  # 低频疾病目标样本数
            'rare_freq': 1500,  # 罕见疾病目标样本数
            'min_samples': 100  # 最小样本数阈值
        }

    def extract_diagnosis_from_output(self, output_text: str) -> str:
        """从output字段中提取诊断信息"""
        if pd.isna(output_text) or output_text == 'nan':
            return "未知诊断"

        patterns = [
            r'主要诊断[：:]\s*([^\n\r]+)',
            r'诊断类别[：:]\s*([^\n\r]+)',
            r'诊断[：:]\s*([^\n\r]+)',
            r'初步诊断[：:]\s*([^\n\r]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, str(output_text))
            if match:
                diagnosis = match.group(1).strip()
                diagnosis = re.sub(r'[，,；;。.].*$', '', diagnosis)
                return diagnosis

        lines = [line.strip() for line in str(output_text).split('\n') if line.strip()]
        if lines:
            first_line = lines[0]
            if '：' in first_line or ':' in first_line:
                parts = re.split('[：:]', first_line, 1)
                if len(parts) > 1:
                    return parts[1].strip()
            return first_line

        return "未知诊断"

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载JSON数据文件"""
        logger.info(f"正在加载数据文件: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # 从output字段提取诊断信息
        if 'output' in df.columns:
            df['diagnosis'] = df['output'].apply(self.extract_diagnosis_from_output)
            logger.info("已从output字段提取诊断信息")

        logger.info(f"成功加载数据，共 {len(df)} 条记录")
        return df

    def analyze_data_distribution(self, df: pd.DataFrame) -> dict:
        """分析数据分布"""
        logger.info("开始分析数据分布...")

        disease_counts = df['diagnosis'].value_counts()
        total_samples = len(df)
        n_diseases = len(disease_counts)

        max_count = disease_counts.max()
        min_count = disease_counts.min()
        imbalance_ratio = max_count / min_count

        print("\n" + "=" * 60)
        print("数据分布分析结果")
        print("=" * 60)
        print(f"总样本数: {total_samples:,}")
        print(f"疾病类别数: {n_diseases}")
        print(f"不平衡比例: {imbalance_ratio:.2f}:1")
        print(f"最高频疾病: {disease_counts.index[0]} ({disease_counts.iloc[0]:,} 样本)")
        print(f"最低频疾病: {disease_counts.index[-1]} ({disease_counts.iloc[-1]:,} 样本)")

        print(f"\n前10个疾病分布:")
        for i, (disease, count) in enumerate(disease_counts.head(10).items()):
            percentage = (count / total_samples) * 100
            print(f"{i + 1:2d}. {disease}: {count:,} ({percentage:.2f}%)")

        return {
            'total_samples': total_samples,
            'n_diseases': n_diseases,
            'disease_counts': disease_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }

    def categorize_diseases(self, disease_counts: dict) -> dict:
        """根据样本数量将疾病分类"""
        categories = {
            'high_freq': [],  # >30k
            'mid_freq': [],  # 5k-30k
            'low_freq': [],  # 1k-5k
            'rare_freq': []  # <1k但>100
        }

        for disease, count in disease_counts.items():
            if count > 30000:
                categories['high_freq'].append(disease)
            elif count > 5000:
                categories['mid_freq'].append(disease)
            elif count > 1000:
                categories['low_freq'].append(disease)
            elif count >= self.target_samples['min_samples']:
                categories['rare_freq'].append(disease)

        return categories

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """平衡数据集"""
        logger.info("开始平衡数据集...")

        disease_counts = df['diagnosis'].value_counts().to_dict()
        disease_categories = self.categorize_diseases(disease_counts)

        balanced_parts = []

        for category, diseases in disease_categories.items():
            if not diseases:
                continue

            target_size = self.target_samples[category]
            logger.info(f"处理 {category} 类别，目标样本数: {target_size}")

            for disease in diseases:
                disease_df = df[df['diagnosis'] == disease].copy()
                current_size = len(disease_df)

                logger.info(f"  {disease}: {current_size} -> {target_size}")

                if current_size > target_size:
                    # 下采样
                    sampled_df = disease_df.sample(n=target_size, random_state=self.random_seed)
                    balanced_parts.append(sampled_df)
                elif current_size < target_size:
                    # 上采样
                    upsampled_df = resample(disease_df,
                                            n_samples=target_size,
                                            random_state=self.random_seed,
                                            replace=True)
                    balanced_parts.append(upsampled_df)
                else:
                    balanced_parts.append(disease_df)

        balanced_df = pd.concat(balanced_parts, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        logger.info(f"数据平衡完成！原始: {len(df)} -> 平衡后: {len(balanced_df)}")
        return balanced_df

    def calculate_class_weights(self, disease_counts: dict) -> dict:
        """计算类别权重"""
        total_samples = sum(disease_counts.values())
        n_classes = len(disease_counts)

        class_weights = {}
        for disease, count in disease_counts.items():
            weight = total_samples / (n_classes * count)
            class_weights[disease] = round(weight, 4)

        print("\n类别权重（前10个最高权重）:")
        sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
        for i, (disease, weight) in enumerate(sorted_weights[:10]):
            print(f"{i + 1:2d}. {disease}: {weight:.4f}")

        return class_weights

    def save_data(self, train_df: pd.DataFrame, class_weights: dict,
                  output_dir: str = "./balanced_medical_data"):
        """保存处理后的数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存训练数据
        train_df.to_json(output_path / "train.jsonl", orient='records', lines=True, force_ascii=False)

        # 保存类别权重
        with open(output_path / "class_weights.json", 'w', encoding='utf-8') as f:
            json.dump(class_weights, f, ensure_ascii=False, indent=2)

        # 保存配置信息
        config = {
            "train_samples": len(train_df),
            "total_samples": len(train_df),
            "n_classes": train_df['diagnosis'].nunique()
        }

        with open(output_path / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info(f"数据已保存到: {output_path}")
        print(f"\n数据保存完成:")
        print(f"{output_path}/")
        print(f"  train.jsonl ({len(train_df)} 条)")
        print(f"  class_weights.json")
        print(f"  dataset_info.json")


def main():
    """主函数"""
    print("医疗数据平衡处理工具")
    print("=" * 60)

    # 配置参数 - 在这里修改你的文件路径
    INPUT_FILE = "/root/autodl-tmp/LLaMA-Factory/data/train.json"
    OUTPUT_DIR = "/root/autodl-tmp/LLaMA-Factory/data/balanced_medical_data"

    try:
        processor = MedicalDataProcessor(random_seed=42)

        # 1. 加载数据
        df = processor.load_data(INPUT_FILE)

        # 2. 检查数据质量
        print(f"\n数据质量检查:")
        print(f"  字段: {list(df.columns)}")
        null_diagnosis = df['diagnosis'].isnull().sum()
        print(f"  空诊断数: {null_diagnosis}")
        if null_diagnosis > 0:
            df = df.dropna(subset=['diagnosis'])
        print(f"  有效样本: {len(df)}")

        # 3. 分析分布
        analysis = processor.analyze_data_distribution(df)

        # 4. 计算权重
        class_weights = processor.calculate_class_weights(analysis['disease_counts'])

        # 5. 平衡数据
        balanced_df = processor.balance_dataset(df)

        # 6. 分析平衡后分布
        print("\n平衡后数据分布:")
        processor.analyze_data_distribution(balanced_df)

        # 7. 保存数据
        processor.save_data(balanced_df, class_weights, OUTPUT_DIR)

        print("\n处理完成！")

    except Exception as e:
        print(f"错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()