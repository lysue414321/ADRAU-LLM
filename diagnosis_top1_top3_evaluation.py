"""
LLM 诊断性能评估 —— 双文件版本(v4)

适用场景:
  - top-1 和 top-3 用的是两个不同的 jsonl 文件(两次独立推理,不同 prompt)
  - top-1 文件:predict 只输出 1 个诊断 (主要诊断 + 疾病编码)
  - top-3 文件:predict 输出 3 个诊断 (主要诊断/次要诊断/第三诊断)

用法:
  1. 修改下面"配置区"里的 JSONL_PATH_TOP1 和 JSONL_PATH_TOP3
  2. 确保 disease_en_mapping.json 和 name2code_mapping.json 在脚本同目录
  3. pip install scikit-learn pandas matplotlib seaborn
  4. python llm_diagnosis_eval.py

输出(./eval_results/ 目录):
  - overall_metrics.csv            Top-1/Top-3 整体指标
  - per_disease_metrics.csv        每个疾病的 top-1/top-3 precision/recall/F1(中英双语)
  - confusion_matrix_full.png      全量混淆矩阵(基于 top-1 文件)
  - confusion_matrix_top10.png     Top 10 混淆矩阵(基于 top-1 文件)
  - prediction_details_top1.csv    Top-1 文件的预测明细
  - prediction_details_top3.csv    Top-3 文件的预测明细
  - top_error_pairs.csv            Top-1 最常见的错误预测对(中英双语)
"""

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ==================== 配置区(改这里) ====================
JSONL_PATH_TOP1 = r"C:\Users\38674\Desktop\23-24\23-24balanced-eva\置信度最新.jsonl"
JSONL_PATH_TOP3 = r"C:\Users\38674\Desktop\23-24\23-24balanced-eva\top3准确率.jsonl"

MAPPING_PATH = "name2code_mapping.json"          # 中文病名→编码
EN_MAPPING_PATH = "disease_en_mapping.json"      # 中文病名→英文
OUTPUT_DIR = "./eval_results"

ENABLE_FUZZY_MATCH = True
MATPLOTLIB_FONT = 'Arial'                        # 图全英文

TOP_N_DISEASES = 10
# =========================================================


INVALID = "__INVALID__"

PREFIX_PATTERN = re.compile(
    r'^\s*(主要诊断|第一诊断|首要诊断|次要诊断|第二诊断|次诊断|第三诊断|三诊断|诊断[123一二三]?)\s*[:：]\s*(.+?)\s*$'
)
LABEL_NAME_PATTERN = re.compile(r'主要诊断\s*[:：]\s*(.+?)\s*$', re.MULTILINE)
LABEL_CODE_PATTERN = re.compile(r'疾病编码\s*[:：]\s*(.+?)\s*$', re.MULTILINE)


# ==================== 映射表 ====================

def build_mapping_from_labels(jsonl_paths, output_path):
    """扫描一个或多个 jsonl,从 label 构造病名→编码映射表"""
    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]
    print(f"  [未找到映射表,从 label 自动生成]")
    name2code_cnt = defaultdict(Counter)
    for jp in jsonl_paths:
        if not os.path.exists(jp):
            continue
        with open(jp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label_text = rec.get('label', '') or ''
                name_m = LABEL_NAME_PATTERN.search(label_text)
                code_m = LABEL_CODE_PATTERN.search(label_text)
                if name_m and code_m:
                    name2code_cnt[name_m.group(1).strip()][code_m.group(1).strip()] += 1
    mapping = {name: codes.most_common(1)[0][0] for name, codes in name2code_cnt.items()}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  [已生成映射表 {output_path},共 {len(mapping)} 条]")
    return mapping


def load_or_build_mapping(mapping_path, jsonl_paths):
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"  已加载映射表 {mapping_path},共 {len(mapping)} 条")
        return mapping
    return build_mapping_from_labels(jsonl_paths, mapping_path)


def load_en_mapping(path):
    if not os.path.exists(path):
        print(f"  [WARN] 未找到 {path},图表将使用 ICD 编码")
        return {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    en_full, en_short = {}, {}
    for k, v in raw.items():
        if k.startswith('_'):
            continue
        if isinstance(v, dict):
            en_full[k] = v.get('en_full', k)
            en_short[k] = v.get('en_short', v.get('en_full', k))
    print(f"  已加载英文映射 {path},共 {len(en_full)} 条")
    return en_full, en_short


# ==================== 病名解析与映射 ====================

def normalize_name(name):
    if not name:
        return ""
    return name.strip().replace('(', '(').replace(')', ')')


def remove_parens(name):
    s = normalize_name(name)
    return re.sub(r'\(.*?\)', '', s).strip()


def match_disease_to_code(name, name2code, enable_fuzzy=True):
    if not name:
        return (INVALID, "empty")
    if name in name2code:
        return (name2code[name], "exact")
    norm = normalize_name(name)
    if norm in name2code:
        return (name2code[norm], "normalized")
    if not enable_fuzzy:
        return (INVALID, "no_exact_match")
    name_noparen = remove_parens(name)
    if name_noparen:
        for candidate, code in name2code.items():
            if remove_parens(candidate) == name_noparen:
                return (code, "parens_removed")
    m = re.match(r'^(.+?)伴有(.+)$', name)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        swapped = f"{b}伴有{a}"
        if swapped in name2code:
            return (name2code[swapped], "swapped_concat")
    return (INVALID, "no_match")


def parse_predict(predict_text, name2code, enable_fuzzy=True, max_k=3):
    """从 predict 文本按顺序提取前 max_k 个预测"""
    if not predict_text:
        return []
    preds = []
    for line in predict_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = PREFIX_PATTERN.match(line)
        if not m:
            continue
        raw_name = m.group(2).strip()
        code, method = match_disease_to_code(raw_name, name2code, enable_fuzzy)
        preds.append((code, raw_name, method))
        if len(preds) >= max_k:
            break
    return preds


def parse_label(label_text):
    name_match = LABEL_NAME_PATTERN.search(label_text or "")
    code_match = LABEL_CODE_PATTERN.search(label_text or "")
    return (name_match.group(1).strip() if name_match else None,
            code_match.group(1).strip() if code_match else None)


def build_code_to_display_name_zh(name2code):
    code_to_names = defaultdict(list)
    for name, code in name2code.items():
        code_to_names[code].append(name)
    return {code: min(names, key=len) for code, names in code_to_names.items()}


def build_code_to_display_name_en(name2code, zh_to_en):
    code_to_names = defaultdict(list)
    for name, code in name2code.items():
        code_to_names[code].append(name)
    result = {}
    for code, names in code_to_names.items():
        rep_zh = min(names, key=len)
        result[code] = zh_to_en.get(rep_zh, rep_zh)
    return result


# ==================== 数据加载 ====================

def load_and_parse(jsonl_path, name2code, enable_fuzzy=True, max_k=3):
    """读取 jsonl,每条输出 (true_code, pred_codes_ordered, 等等)"""
    records = []
    match_counter = Counter()
    line_counter = Counter()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 第 {idx} 行 JSON 解析失败,跳过")
                continue
            label_text = rec.get('label', '') or ''
            predict_text = rec.get('predict', '') or ''
            true_name, true_code = parse_label(label_text)
            if true_code is None:
                continue
            preds = parse_predict(predict_text, name2code, enable_fuzzy, max_k=max_k)
            pred_codes = [p[0] for p in preds]
            pred_names = [p[1] for p in preds]
            for _, _, method in preds:
                match_counter[method] += 1
            top1 = pred_codes[0] if pred_codes else INVALID
            topk_set = set(pred_codes) if pred_codes else {INVALID}
            raw_line_count = sum(1 for l in predict_text.splitlines() if l.strip())
            line_counter[raw_line_count] += 1
            records.append({
                'idx': idx,
                'true_code': true_code,
                'true_name': true_name,
                'top1_code': top1,
                'topk_codes': list(topk_set),
                'pred_codes_ordered': pred_codes,
                'pred_names_ordered': pred_names,
                'raw_line_count': raw_line_count,
            })
    return records, match_counter, line_counter


# ==================== 指标计算 ====================

def compute_top1_accuracy(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r['top1_code'] == r['true_code']) / len(records)


def compute_topk_accuracy(records):
    """真实编码在 topk_codes 集合里就算命中"""
    if not records:
        return 0.0
    return sum(1 for r in records if r['true_code'] in r['topk_codes']) / len(records)


def build_top1_vectors(records):
    return [r['true_code'] for r in records], [r['top1_code'] for r in records]


def build_topk_vectors(records):
    """Top-k 下:真实编码在 topk_codes 里 → 记为正确预测;否则归因到 top-1"""
    y_true, y_pred = [], []
    for r in records:
        t = r['true_code']
        if t in r['topk_codes']:
            y_true.append(t)
            y_pred.append(t)
        else:
            y_true.append(t)
            y_pred.append(r['top1_code'])
    return y_true, y_pred


def compute_overall_metrics(y_true, y_pred, valid_codes, tag):
    acc = accuracy_score(y_true, y_pred)
    result = {'tag': tag, 'accuracy': acc}
    for avg in ['macro', 'weighted', 'micro']:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=valid_codes, average=avg, zero_division=0
        )
        result[f'precision_{avg}'] = p
        result[f'recall_{avg}'] = r
        result[f'f1_{avg}'] = f
    return result


def compute_per_disease_table(records_top1, records_top3, valid_codes, code2zh, code2en_full):
    """Top-1 来自 top1 文件,Top-3 来自 top3 文件,分别算后合并"""
    y1_true, y1_pred = build_top1_vectors(records_top1)
    y3_true, y3_pred = build_topk_vectors(records_top3)

    p1, r1, f1, s1 = precision_recall_fscore_support(
        y1_true, y1_pred, labels=valid_codes, average=None, zero_division=0
    )
    p3, r3, f3, s3 = precision_recall_fscore_support(
        y3_true, y3_pred, labels=valid_codes, average=None, zero_division=0
    )

    rows = []
    for i, code in enumerate(valid_codes):
        rows.append({
            'code': code,
            'name_zh': code2zh.get(code, code),
            'name_en': code2en_full.get(code, code),
            'support_top1': int(s1[i]),
            'support_top3': int(s3[i]),
            'top1_precision': round(float(p1[i]), 4),
            'top1_recall': round(float(r1[i]), 4),
            'top1_f1': round(float(f1[i]), 4),
            'top3_precision': round(float(p3[i]), 4),
            'top3_recall': round(float(r3[i]), 4),
            'top3_f1': round(float(f3[i]), 4),
        })
    return pd.DataFrame(rows).sort_values('support_top1', ascending=False).reset_index(drop=True)


# ==================== 可视化 ====================

def plot_full_confusion_matrix(y_true, y_pred, valid_codes, code2en_short, output_path):
    support_cnt = Counter(y_true)
    sorted_codes = sorted(valid_codes, key=lambda c: -support_cnt.get(c, 0))

    cm = confusion_matrix(y_true, y_pred, labels=sorted_codes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    labels = [f"{code}\n{code2en_short.get(code, '')}" for code in sorted_codes]

    n = len(sorted_codes)
    size = max(24, int(n * 0.75))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        cm_norm, annot=cm, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Row-normalized proportion'},
        annot_kws={'size': 9},
        linewidths=0.3, linecolor='gray',
        ax=ax,
    )
    ax.set_xlabel('Predicted diagnosis (ICD code)', fontsize=14)
    ax.set_ylabel('True diagnosis (ICD code)', fontsize=14)
    ax.set_title(
        f'Top-1 Confusion Matrix (all {n} disease codes, sorted by support)',
        fontsize=16,
    )
    plt.xticks(rotation=60, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_topn_confusion_matrix(y_true, y_pred, valid_codes, code2en_short, n, output_path):
    support_cnt = Counter(y_true)
    topn = [c for c, _ in support_cnt.most_common(n)]
    topn_set = set(topn)

    filt_y_true, filt_y_pred = [], []
    for t, p in zip(y_true, y_pred):
        if t in topn_set:
            filt_y_true.append(t)
            filt_y_pred.append(p if p in topn_set else 'Other')

    labels_order = topn + ['Other']
    cm = confusion_matrix(filt_y_true, filt_y_pred, labels=labels_order)

    display_labels = [f"{code}\n{code2en_short.get(code, '')}" for code in topn] + ['Other']

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm, annot=cm, fmt='d', cmap='Blues',
        xticklabels=display_labels, yticklabels=display_labels,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 11},
        linewidths=0.5, linecolor='gray',
        ax=ax,
    )
    ax.set_xlabel('Predicted diagnosis', fontsize=13)
    ax.set_ylabel('True diagnosis', fontsize=13)
    ax.set_title(
        f'Top-1 Confusion Matrix — Top {n} diseases (plus Other)',
        fontsize=15,
    )
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# ==================== 数据集级别统计输出 ====================

def print_dataset_stats(records, match_counter, line_counter, tag):
    print(f"\n  [{tag} 文件统计]")
    print(f"    有效样本数: {len(records)}")
    print(f"    匹配方式分布: {dict(match_counter)}")
    abnormal = sum(c for n, c in line_counter.items() if n > 5)
    print(f"    生成格式异常(输出>5行): {abnormal} ({abnormal/len(records)*100:.2f}%)")
    n_inv = sum(1 for r in records if r['top1_code'] == INVALID)
    print(f"    Top-1 无效预测: {n_inv} ({n_inv/len(records)*100:.2f}%)")
    n_same = sum(1 for r in records if len(r['pred_codes_ordered']) >= 2
                 and len(set(r['pred_codes_ordered'])) == 1)
    if any(len(r['pred_codes_ordered']) >= 2 for r in records):
        print(f"    预测完全相同(编码层面): {n_same} ({n_same/len(records)*100:.2f}%)")


# ==================== 主流程 ====================

def main():
    try:
        matplotlib.rcParams['font.sans-serif'] = [MATPLOTLIB_FONT, 'DejaVu Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"[WARN] 字体设置失败: {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 映射表
    print("=" * 70)
    print("[1/5] 加载映射表")
    name2code = load_or_build_mapping(MAPPING_PATH, [JSONL_PATH_TOP1, JSONL_PATH_TOP3])
    zh_to_en_full, zh_to_en_short = load_en_mapping(EN_MAPPING_PATH)

    valid_codes = sorted(set(name2code.values()))
    code2zh = build_code_to_display_name_zh(name2code)
    code2en_full = build_code_to_display_name_en(name2code, zh_to_en_full)
    code2en_short = build_code_to_display_name_en(name2code, zh_to_en_short)
    print(f"  有效编码数(去重): {len(valid_codes)}")

    missing_en = [c for c in valid_codes if code2en_full.get(c) == code2zh.get(c)]
    if missing_en:
        print(f"  [WARN] 以下编码缺少英文映射,将回退到中文:")
        for c in missing_en:
            print(f"    {c}: {code2zh.get(c)}")

    # 2. 解析两个文件
    print("\n" + "=" * 70)
    print("[2/5] 解析 jsonl 文件")
    print(f"  Top-1 文件: {JSONL_PATH_TOP1}")
    records_top1, mc1, lc1 = load_and_parse(
        JSONL_PATH_TOP1, name2code, enable_fuzzy=ENABLE_FUZZY_MATCH, max_k=1
    )
    print_dataset_stats(records_top1, mc1, lc1, 'Top-1')

    print(f"\n  Top-3 文件: {JSONL_PATH_TOP3}")
    records_top3, mc3, lc3 = load_and_parse(
        JSONL_PATH_TOP3, name2code, enable_fuzzy=ENABLE_FUZZY_MATCH, max_k=3
    )
    print_dataset_stats(records_top3, mc3, lc3, 'Top-3')

    # 3. 整体指标
    print("\n" + "=" * 70)
    print("[3/5] 计算整体指标")

    top1_acc = compute_top1_accuracy(records_top1)
    top3_acc = compute_topk_accuracy(records_top3)
    print(f"  Top-1 Accuracy (from top-1 file): {top1_acc:.4f}")
    print(f"  Top-3 Accuracy (from top-3 file): {top3_acc:.4f}")

    y_true_1, y_pred_1 = build_top1_vectors(records_top1)
    y_true_3, y_pred_3 = build_topk_vectors(records_top3)
    m1 = compute_overall_metrics(y_true_1, y_pred_1, valid_codes, 'top-1')
    m3 = compute_overall_metrics(y_true_3, y_pred_3, valid_codes, 'top-3')

    print("\n  [Top-1 Overall]")
    for k, v in m1.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print("\n  [Top-3 Overall]")
    for k, v in m3.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    pd.DataFrame([m1, m3]).to_csv(
        os.path.join(OUTPUT_DIR, 'overall_metrics.csv'),
        index=False, encoding='utf-8-sig'
    )

    # 4. Per-disease + 混淆矩阵(混淆矩阵只基于 top-1 文件)
    print("\n" + "=" * 70)
    print("[4/5] 计算 per-disease 指标 + 绘制混淆矩阵")
    per_df = compute_per_disease_table(records_top1, records_top3,
                                         valid_codes, code2zh, code2en_full)
    per_df.to_csv(os.path.join(OUTPUT_DIR, 'per_disease_metrics.csv'),
                  index=False, encoding='utf-8-sig')
    print(f"  per-disease 指标已保存,共 {len(per_df)} 行")

    cm_records = [(r['true_code'], r['top1_code'])
                  for r in records_top1 if r['top1_code'] != INVALID]
    y_cm = [t for t, _ in cm_records]
    p_cm = [p for _, p in cm_records]

    plot_full_confusion_matrix(
        y_cm, p_cm, valid_codes, code2en_short,
        os.path.join(OUTPUT_DIR, 'confusion_matrix_full.png')
    )
    print(f"  全量混淆矩阵已保存")

    plot_topn_confusion_matrix(
        y_cm, p_cm, valid_codes, code2en_short, TOP_N_DISEASES,
        os.path.join(OUTPUT_DIR, 'confusion_matrix_top10.png')
    )
    print(f"  Top-{TOP_N_DISEASES} 混淆矩阵已保存")

    # 5. 明细 + 错误分析
    print("\n" + "=" * 70)
    print("[5/5] 导出预测明细与错误分析")

    # Top-1 明细
    details_top1 = pd.DataFrame([{
        'idx': r['idx'],
        'true_code': r['true_code'],
        'true_name_zh': r['true_name'],
        'pred_code': r['top1_code'],
        'pred_name_zh': ';'.join(r['pred_names_ordered']),
        'hit': int(r['top1_code'] == r['true_code']),
    } for r in records_top1])
    details_top1.to_csv(os.path.join(OUTPUT_DIR, 'prediction_details_top1.csv'),
                         index=False, encoding='utf-8-sig')

    # Top-3 明细
    details_top3 = pd.DataFrame([{
        'idx': r['idx'],
        'true_code': r['true_code'],
        'true_name_zh': r['true_name'],
        'top1_code': r['top1_code'],
        'top3_codes': ';'.join(r['topk_codes']),
        'pred_names_zh': ';'.join(r['pred_names_ordered']),
        'top1_hit': int(r['top1_code'] == r['true_code']),
        'top3_hit': int(r['true_code'] in r['topk_codes']),
        'raw_line_count': r['raw_line_count'],
    } for r in records_top3])
    details_top3.to_csv(os.path.join(OUTPUT_DIR, 'prediction_details_top3.csv'),
                         index=False, encoding='utf-8-sig')

    # Top-1 错误分析
    err_pairs = Counter((r['true_code'], r['top1_code'])
                        for r in records_top1 if r['top1_code'] != r['true_code'])
    print("\n  [Top-1] 错误最多的 真实→预测 对 (Top 20):")
    err_rows = []
    for (tc, pc), cnt in err_pairs.most_common(20):
        pn_zh = code2zh.get(pc, pc) if pc != INVALID else INVALID
        pn_en = code2en_full.get(pc, pc) if pc != INVALID else INVALID
        err_rows.append({
            'true_code': tc,
            'true_name_zh': code2zh.get(tc, tc),
            'true_name_en': code2en_full.get(tc, tc),
            'pred_code': pc,
            'pred_name_zh': pn_zh,
            'pred_name_en': pn_en,
            'count': cnt,
        })
        print(f"    {code2zh.get(tc, tc)}({tc}) → {pn_zh}({pc}): {cnt}")
    pd.DataFrame(err_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'top_error_pairs.csv'),
        index=False, encoding='utf-8-sig'
    )

    print("\n" + "=" * 70)
    print(f"完成。所有结果在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()