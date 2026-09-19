import re
import pandas as pd

# 医学参考值字典：包含范围、单位、中文说明
normal_ranges = {
    "WBC": (3.5, 9.5, "E+9/L", "白细胞总数"),
    "NEU#": (1.8, 6.3, "E+9/L", "中性粒细胞绝对值"),
    "NEU%": (40, 75, "%", "中性粒细胞百分比"),
    "LYM%": (20, 50, "%", "淋巴细胞百分比"),
    "LYM#": (1.1, 3.2, "E+9/L", "淋巴细胞绝对值"),
    "MONO%": (3, 10, "%", "单核细胞百分比"),
    "MONO#": (0.1, 0.6, "E+9/L", "单核细胞绝对值"),
    "EO#": (0.02, 0.52, "E+9/L", "嗜酸性粒细胞绝对值"),
    "EOS#": (0.02, 0.52, "E+9/L", "嗜酸性粒细胞绝对值"),
    "BASO#": (0, 0.06, "E+9/L", "嗜碱性粒细胞绝对值"),
    "EO%": (0.4, 8, "%", "嗜酸性粒细胞百分比"),
    "EOS%": (0.4, 8, "%", "嗜酸性粒细胞百分比"),
    "BASO%": (0, 1, "%", "嗜碱性粒细胞百分比"),
    "RBC": (3.8, 5.8, "E+12/L", "红细胞计数"),
    "HGB": (115, 175, "g/L", "血红蛋白"),
    "HCT": (35, 50, "%", "红细胞压积"),
    "MCV": (82, 100, "fL", "平均红细胞体积"),
    "MCH": (27, 34, "Pg", "平均血红蛋白含量"),
    "MCHC": (316, 354, "g/L", "平均血红蛋白浓度"),
    "PLT": (125, 350, "E+9/L", "血小板计数"),
    "MPV": (6.2, 10.9, "fL", "平均血小板体积"),
    "PDW": (16, 21, "fL", "血小板分布宽度"),
    "P-LCR": (13, 43, "%", "大血小板比率"),
    "PCT": (0.1, 0.28, "%", "血小板压积"),
    "NRBC#": (0, 0, "E+9/L", "有核红细胞绝对值"),
    "NRBC%": (0, 0, "%", "有核红细胞百分比"),
    "RDW-CV": (11.6, 13.7, "%", "红细胞分布宽度-CV"),
    "RDW-SD": (41.8, 45.8, "fL", "红细胞分布宽度-SD"),
    "RET_L": (85.55, 98.3, "%", "红细胞生成-低荧光强度"),
    "RET_M": (1.6, 11.8, "%", "红细胞生成-中荧光强度"),
    "RET_H": (0, 2.11, "%", "红细胞生成-高荧光强度"),
    "Total-IgE": (0.35, 100, "kU/L", "总免疫球蛋白E"),
    "IGE": (0.35, 100, "IU/L", "免疫球蛋白E"),
    "FENO": (0, 25, "ppb", "呼出气一氧化氮"),
    "FNNO": (74, 84, "ppb", "鼻呼气一氧化氮"),
    "hs-CRP": (0, 5, "mg/L", "超敏C反应蛋白"),
    "BO2": (18, 24, "mmol/L", "总血氧含量"),
    "SO2": (93.4, 98.5, "%", "血氧饱和度"),
    "FO2Hb": (90, 97, "%", "氧合血红蛋白"),
    "FCOHb": (0, 2, "%", "一氧化碳血红蛋白"),
    "FMetHb": (0, 1, "%", "高铁血红蛋白"),
    "FHHb": (1, 3, "%", "还原血红蛋白"),
    "pH(T)": (7.35, 7.45, "", "动脉血pH值"),
    "pCO2(T)": (35, 45, "mmHg", "动脉二氧化碳分压"),
    "pO2(T)": (95, 100, "mmHg", "动脉氧分压"),
    "pO2(A-a)(T)": (15, 20, "mmHg", "肺泡-动脉氧差"),
    "pO2(a/A)(T)": (0.75, 1.0, "", "动脉/肺泡氧比值"),
    "RI(T)": (0, 1.0, "", "呼吸指数"),
    "Temp": (36.5, 37.5, "℃", "体温"),
    "FIO2": (0.21, None, "%", "吸入氧浓度"),
    "LAC": (0.5, 2.0, "mmol/L", "乳酸"),
    "pH": (7.35, 7.45, "", "血液pH"),
    "PCO2": (35, 45, "mmHg", "二氧化碳分压"),
    "PO2": (80, 100, "mmHg", "氧分压"),
    "HCO3-act": (22, 27, "mmol/L", "实际碳酸氢盐"),
    "ctCO2": (23, 28, "mmol/L", "总二氧化碳"),
    "HCO3-std": (22, 26, "mmol/L", "标准碳酸氢盐"),
    "BE(B)": (-2.3, 2.3, "mmol/L", "碱剩余（血液）"),
    "BE(ecf)": (-2.3, 2.3, "mmol/L", "碱剩余（细胞外液）"),
    "ctHb": (120, 160, "g/L", "总血红蛋白"),
    # 心肌标志物
    "c-TnI": (0, 0.034, "ng/mL", "肌钙蛋白I"),
    "cTnT": (0, 0.01, "ng/mL", "肌钙蛋白T"),
    "CK": (25, 200, "U/L", "肌酸激酶"),
    "CKMB": (0, 24, "U/L", "肌酸激酶同工酶"),
    "MB/CK": (0.007, 0.063, "", "MB占CK比例"),
    # 电解质
    "Na+": (137, 147, "mmol/L", "钠离子"),
    "K+": (3.5, 5.3, "mmol/L", "钾离子"),
    "Ca++": (2.53, 3.50, "mmol/L", "游离钙"),
    # 肺功能指标
    "FEV1/FVC": (80, 100, "%", "一秒率"),
    "FEV1/PRE": (80, 120, "%", "FEV1占预计值百分比"),
    # 溶血指数
    "溶血指数": (0, 20, "", "溶血指数")
}

# 按照关键词排序，确保长词优先匹配（如 NEU# 在 NEU 前）
sorted_keywords = sorted(normal_ranges.keys(), key=len, reverse=True)


def process_item(text_part):
    """
    对每一个分项（如 'RDW-CV 14.1%'）进行处理，
    如果匹配到某个关键字（如 RDW-CV），就在其后插入参考值信息。
    """
    text_part = text_part.strip()

    for keyword in sorted_keywords:
        # 构建精确匹配的正则表达式，支持 = 或空格分隔
        pattern = rf"({re.escape(keyword)})[= ]([^=]+)"
        match = re.search(pattern, text_part, re.IGNORECASE)

        if match:
            raw_key, value = match.groups()
            info = normal_ranges.get(keyword)
            if info and isinstance(info, tuple) and len(info) == 4:
                low, high, unit, name = info
                range_str = f"{low}-{high}" if low is not None and high is not None else ""
                unit_str = f" {unit}" if unit else ""

                replacement = f"{raw_key}（{name}：正常范围 {range_str}{unit_str}） {value}"
                new_text = re.sub(pattern, replacement, text_part, flags=re.IGNORECASE)
                return new_text

    return text_part


def add_normal_range(text):
    if not isinstance(text, str):
        return text

    # 支持多种分隔符
    parts = re.split(r'[、；,]', text)

    processed_parts = []
    for part in parts:
        processed_part = process_item(part.strip())
        processed_parts.append(processed_part)

    return '、'.join(processed_parts)


# 读取Excel文件
file_path = 'C:/Users/38674/Desktop/23-24.xlsx'
output_path = 'C:/Users/38674/Desktop/23-24年最终版.xlsx'

try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"❌ 无法读取Excel文件：{e}")
    exit()

# 处理指定列，未匹配字段将保持不变
if '辅助检查' in df.columns:
    df['辅助检查'] = df['辅助检查'].apply(add_normal_range)
else:
    print("⚠️ 注意：列名‘辅助检查’不存在，程序已跳过处理。")

# 保存结果到新文件
try:
    df.to_excel(output_path, index=False)
    print(f"✅ 处理完成！输出文件：{output_path}")
except Exception as e:
    print(f"❌ 无法写入Excel文件：{e}")