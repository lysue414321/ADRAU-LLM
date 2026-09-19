import rdflib
import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import re

# 输入文件路径
owl_file_path = r"/antibiotic_change_IRI.owl"  # 替换为您的OWL文件路径
output_dir = "output11"  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)


def fix_owl_file(input_path, output_path):
    """尝试修复OWL文件的常见问题"""
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 替换可能导致问题的XML模式
    # 修复重复的IRI节点问题
    fixed_content = content.replace('<IRI><IRI>', '<IRI>')
    fixed_content = fixed_content.replace('</IRI></IRI>', '</IRI>')

    # 处理XML命名空间问题
    if '&xsd;' in fixed_content and '<!DOCTYPE' not in fixed_content:
        # 添加DTD声明
        dtd_decl = '<!DOCTYPE rdf:RDF [\n  <!ENTITY xsd "http://www.w3.org/2001/XMLSchema#">\n]>\n'
        if '<?xml' in fixed_content:
            xml_end = fixed_content.find('?>') + 2
            fixed_content = fixed_content[:xml_end] + '\n' + dtd_decl + fixed_content[xml_end:]
        else:
            fixed_content = dtd_decl + fixed_content

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"已创建修复版本: {output_path}")
    return output_path


# 使用修复函数
fixed_owl_path = os.path.join(output_dir, "fixed_owl.owl")
fixed_path = fix_owl_file(owl_file_path, fixed_owl_path)

# 尝试不同的解析方法
g = rdflib.Graph()
successfully_parsed = False

try:
    print("尝试解析修复后的OWL文件...")
    g.parse(fixed_path, format="xml")
    successfully_parsed = True
    print("成功解析修复后的OWL文件")
except Exception as e:
    print(f"修复后解析仍失败: {e}")
    try:
        print("尝试以turtle格式解析原始文件...")
        g.parse(owl_file_path, format="turtle")
        successfully_parsed = True
        print("成功以turtle格式解析")
    except Exception as e:
        print(f"turtle格式解析失败: {e}")
        try:
            print("尝试以n3格式解析...")
            g.parse(owl_file_path, format="n3")
            successfully_parsed = True
            print("成功以n3格式解析")
        except Exception as e:
            print(f"n3格式解析失败: {e}")
            try:
                print("尝试以ntriples格式解析...")
                g.parse(owl_file_path, format="nt")
                successfully_parsed = True
                print("成功以ntriples格式解析")
            except Exception as e:
                print(f"所有标准格式解析失败: {e}")

# 如果标准解析方法都失败，尝试手动解析
if not successfully_parsed:
    print("所有标准解析方法都失败，尝试手动解析...")


    def extract_triples_manually(owl_file_path):
        """手动从OWL文件中提取三元组"""
        triples = []
        entity_labels = {}  # 存储实体ID到标签的映射

        with open(owl_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 查找类定义
        class_pattern = r'<Class IRI="([^"]+)"'
        classes = re.findall(class_pattern, content)
        for class_uri in classes:
            triples.append(
                (class_uri, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://www.w3.org/2002/07/owl#Class"))

        # 查找标签注释
        label_pattern = r'<AnnotationAssertion>\s*<AnnotationProperty IRI="http://www.w3.org/2000/01/rdf-schema#label"/>\s*<IRI>([^<]+)</IRI>\s*<Literal[^>]*>([^<]+)</Literal>\s*</AnnotationAssertion>'
        for match in re.finditer(label_pattern, content, re.DOTALL):
            subject, label = match.groups()
            triples.append((subject, "http://www.w3.org/2000/01/rdf-schema#label", label))
            # 保存标签信息
            subject_id = subject.split("/")[-1].split("#")[-1]
            entity_labels[subject_id] = label

        # 查找注释
        comment_pattern = r'<AnnotationAssertion>\s*<AnnotationProperty IRI="http://www.w3.org/2000/01/rdf-schema#comment"/>\s*<IRI>([^<]+)</IRI>\s*<Literal[^>]*>([^<]+)</Literal>\s*</AnnotationAssertion>'
        for match in re.finditer(comment_pattern, content, re.DOTALL):
            subject, comment = match.groups()
            triples.append((subject, "http://www.w3.org/2000/01/rdf-schema#comment", comment))

        # 查找子类关系
        subclass_pattern = r'<SubClassOf>\s*<Class IRI="([^"]+)"/>\s*<Class IRI="([^"]+)"/>\s*</SubClassOf>'
        for match in re.finditer(subclass_pattern, content, re.DOTALL):
            subclass, superclass = match.groups()
            triples.append((subclass, "http://www.w3.org/2000/01/rdf-schema#subClassOf", superclass))

        # 查找对象属性
        obj_prop_pattern = r'<ObjectProperty IRI="([^"]+)"'
        obj_props = re.findall(obj_prop_pattern, content)
        for prop_uri in obj_props:
            triples.append((prop_uri, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                            "http://www.w3.org/2002/07/owl#ObjectProperty"))

        # 查找数据属性
        data_prop_pattern = r'<DataProperty IRI="([^"]+)"'
        data_props = re.findall(data_prop_pattern, content)
        for prop_uri in data_props:
            triples.append((prop_uri, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                            "http://www.w3.org/2002/07/owl#DatatypeProperty"))

        # 查找对象属性域和值域
        domain_range_pattern = r'<ObjectPropertyDomain>\s*<ObjectProperty IRI="([^"]+)"/>\s*<Class IRI="([^"]+)"/>\s*</ObjectPropertyDomain>'
        for match in re.finditer(domain_range_pattern, content, re.DOTALL):
            prop, domain = match.groups()
            triples.append((prop, "http://www.w3.org/2000/01/rdf-schema#domain", domain))

        range_pattern = r'<ObjectPropertyRange>\s*<ObjectProperty IRI="([^"]+)"/>\s*<Class IRI="([^"]+)"/>\s*</ObjectPropertyRange>'
        for match in re.finditer(range_pattern, content, re.DOTALL):
            prop, range_val = match.groups()
            triples.append((prop, "http://www.w3.org/2000/01/rdf-schema#range", range_val))

        # 查找数据属性域和值域
        data_domain_pattern = r'<DataPropertyDomain>\s*<DataProperty IRI="([^"]+)"/>\s*<Class IRI="([^"]+)"/>\s*</DataPropertyDomain>'
        for match in re.finditer(data_domain_pattern, content, re.DOTALL):
            prop, domain = match.groups()
            triples.append((prop, "http://www.w3.org/2000/01/rdf-schema#domain", domain))

        data_range_pattern = r'<DataPropertyRange>\s*<DataProperty IRI="([^"]+)"/>\s*<Datatype IRI="([^"]+)"/>\s*</DataPropertyRange>'
        for match in re.finditer(data_range_pattern, content, re.DOTALL):
            prop, range_val = match.groups()
            triples.append((prop, "http://www.w3.org/2000/01/rdf-schema#range", range_val))

        # 查找等价类
        equiv_pattern = r'<EquivalentClasses>\s*<Class IRI="([^"]+)"/>\s*<Class IRI="([^"]+)"/>\s*</EquivalentClasses>'
        for match in re.finditer(equiv_pattern, content, re.DOTALL):
            class1, class2 = match.groups()
            triples.append((class1, "http://www.w3.org/2002/07/owl#equivalentClass", class2))

        # 查找不相交类
        disjoint_pattern = r'<DisjointClasses>\s*<Class IRI="([^"]+)"/>\s*<Class IRI="([^"]+)"/>\s*</DisjointClasses>'
        for match in re.finditer(disjoint_pattern, content, re.DOTALL):
            class1, class2 = match.groups()
            triples.append((class1, "http://www.w3.org/2002/07/owl#disjointWith", class2))

        # 查找对象属性断言
        obj_assertion_pattern = r'<ObjectPropertyAssertion>\s*<ObjectProperty IRI="([^"]+)"/>\s*<NamedIndividual IRI="([^"]+)"/>\s*<NamedIndividual IRI="([^"]+)"/>\s*</ObjectPropertyAssertion>'
        for match in re.finditer(obj_assertion_pattern, content, re.DOTALL):
            prop, subject, obj = match.groups()
            triples.append((subject, prop, obj))

        # 查找数据属性断言
        data_assertion_pattern = r'<DataPropertyAssertion>\s*<DataProperty IRI="([^"]+)"/>\s*<NamedIndividual IRI="([^"]+)"/>\s*<Literal[^>]*>([^<]+)</Literal>\s*</DataPropertyAssertion>'
        for match in re.finditer(data_assertion_pattern, content, re.DOTALL):
            prop, subject, value = match.groups()
            triples.append((subject, prop, value))

        # 构建RDFLib图
        g = rdflib.Graph()
        for s, p, o in triples:
            s_node = rdflib.URIRef(s)
            p_node = rdflib.URIRef(p)
            # 判断对象是URI还是字面量
            if o.startswith('http'):
                o_node = rdflib.URIRef(o)
            else:
                o_node = rdflib.Literal(o)
            g.add((s_node, p_node, o_node))

        # 将标签信息保存到图的元数据
        g.store.__entity_labels = entity_labels

        return g


    g = extract_triples_manually(owl_file_path)
    print(f"手动解析完成，提取了 {len(g)} 个三元组")
    successfully_parsed = True

# 如果仍然无法解析，尝试使用owlready2库
if not successfully_parsed:
    try:
        from owlready2 import get_ontology

        print("尝试使用owlready2加载...")
        onto = get_ontology(owl_file_path).load()

        g = rdflib.Graph()
        entity_labels = {}  # 存储实体ID到标签的映射

        # 添加类
        for cls in onto.classes():
            cls_uri = cls.iri
            g.add((rdflib.URIRef(cls_uri), rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                   rdflib.URIRef("http://www.w3.org/2002/07/owl#Class")))

            # 添加标签
            if hasattr(cls, "label") and cls.label:
                label = str(cls.label[0]) if isinstance(cls.label, list) and cls.label else str(cls.label)
                g.add((rdflib.URIRef(cls_uri), rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                       rdflib.Literal(label)))
                cls_id = cls_uri.split("/")[-1].split("#")[-1]
                entity_labels[cls_id] = label

            # 添加注释
            if hasattr(cls, "comment") and cls.comment:
                comment = str(cls.comment[0]) if isinstance(cls.comment, list) and cls.comment else str(cls.comment)
                g.add((rdflib.URIRef(cls_uri), rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#comment"),
                       rdflib.Literal(comment)))

            # 添加子类关系
            for parent in cls.is_a:
                if hasattr(parent, "iri"):
                    g.add((rdflib.URIRef(cls_uri), rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"),
                           rdflib.URIRef(parent.iri)))

        # 添加属性
        for prop in onto.object_properties():
            prop_uri = prop.iri
            g.add((rdflib.URIRef(prop_uri), rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                   rdflib.URIRef("http://www.w3.org/2002/07/owl#ObjectProperty")))

            # 添加域和值域
            if hasattr(prop, "domain") and prop.domain:
                for domain in prop.domain:
                    if hasattr(domain, "iri"):
                        g.add((rdflib.URIRef(prop_uri), rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#domain"),
                               rdflib.URIRef(domain.iri)))

            if hasattr(prop, "range") and prop.range:
                for range_val in prop.range:
                    if hasattr(range_val, "iri"):
                        g.add((rdflib.URIRef(prop_uri), rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#range"),
                               rdflib.URIRef(range_val.iri)))

        # 存储标签信息
        g.store.__entity_labels = entity_labels

        successfully_parsed = True
        print(f"使用owlready2成功加载，转换为RDFLib图包含 {len(g)} 个三元组")
    except Exception as e:
        print(f"owlready2加载失败: {e}")

if not successfully_parsed:
    print("所有解析方法都失败，无法继续处理")
    import sys

    sys.exit(1)

# 提取所有三元组
print("提取三元组...")
triples = []
for s, p, o in tqdm(g):
    s_str = str(s)
    p_str = str(p)
    o_str = str(o)
    triples.append((s_str, p_str, o_str))

print(f"共提取 {len(triples)} 个三元组")

# 创建实体标签映射
entity_labels = getattr(g.store, "__entity_labels", {})

# 如果没有从解析器获取标签，尝试从三元组中提取
if not entity_labels:
    print("从三元组中提取实体标签...")
    for s, p, o in triples:
        if "label" in p.lower():
            s_id = s.split("/")[-1].split("#")[-1]
            entity_labels[s_id] = str(o)

print(f"找到 {len(entity_labels)} 个实体标签")

# 过滤和清洗三元组
print("过滤和清洗三元组...")
filtered_triples = []

# 关系映射，将URI映射到更易理解的名称
relation_mapping = {
    "http://www.w3.org/2000/01/rdf-schema#subClassOf": "subClassOf",
    "http://www.w3.org/2002/07/owl#equivalentClass": "equivalentTo",
    "http://www.w3.org/2002/07/owl#disjointWith": "disjointWith",
    "http://www.w3.org/2000/01/rdf-schema#domain": "hasDomain",
    "http://www.w3.org/2000/01/rdf-schema#range": "hasRange",
    # 添加更多关系映射...
}

# 过滤元数据三元组 (rdf, rdfs, owl命名空间)
metadata_namespaces = [
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
]

# 保留的重要元数据属性
important_props = [
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#disjointWith"
]

for s, p, o in tqdm(triples):
    # 跳过某些元数据相关的三元组，但保留重要的如rdfs:label, rdfs:subClassOf等
    if any(ns in p for ns in metadata_namespaces):
        if not any(p == prop for prop in important_props):
            continue

    # 确保主体是有意义的
    if not (s.startswith("http") or s.startswith("urn:")):
        continue

    # 提取最后一部分作为标签
    s_id = s.split("/")[-1].split("#")[-1]
    p_id = p.split("/")[-1].split("#")[-1]

    # 使用更友好的关系名称
    p_name = relation_mapping.get(p, p_id)

    is_literal = isinstance(o, rdflib.term.Literal) or not (o.startswith("http") or o.startswith("urn:"))

    if is_literal:
        o_id = str(o)
        o_name = o_id
    else:
        o_id = o.split("/")[-1].split("#")[-1]
        o_name = entity_labels.get(o_id, o_id)

    # 使用实体标签替代ID
    s_name = entity_labels.get(s_id, s_id)

    filtered_triples.append((s_id, p_name, o_id, s_name, p_name, o_name, s, p, o, is_literal))

print(f"过滤后剩余 {len(filtered_triples)} 个三元组")

# 创建实体和关系的映射
entities = set()
relations = set()

for s_id, p_name, o_id, s_name, _, o_name, s, p, o, is_literal in filtered_triples:
    entities.add((s_id, s_name, s))
    relations.add((p_name, p))
    if not is_literal:
        entities.add((o_id, o_name, o))

entity_to_id = {entity[0]: idx for idx, entity in enumerate(entities)}
relation_to_id = {rel[0]: idx for idx, rel in enumerate(relations)}

# 保存映射
with open(os.path.join(output_dir, "entity_map.json"), "w", encoding="utf-8") as f:
    json.dump({entity[0]: {
        "id": idx,
        "name": entity[1],
        "uri": entity[2]
    } for idx, entity in enumerate(entities)}, f, indent=2, ensure_ascii=False)

with open(os.path.join(output_dir, "relation_map.json"), "w", encoding="utf-8") as f:
    json.dump({rel[0]: {
        "id": idx,
        "uri": rel[1]
    } for idx, rel in enumerate(relations)}, f, indent=2, ensure_ascii=False)

# 创建知识对数据集，包含文字属性
knowledge_pairs = []

for s_id, p_name, o_id, s_name, _, o_name, s, p, o, is_literal in filtered_triples:
    # 包含实体-关系-实体和实体-属性-值三元组
    knowledge_pairs.append({
        "head": s_name,
        "head_id": entity_to_id[s_id],
        "head_original": s_id,
        "relation": p_name,
        "relation_id": relation_to_id[p_name],
        "tail": o_name,
        "tail_id": entity_to_id.get(o_id, -1) if not is_literal else -1,
        "tail_original": o_id,
        "head_uri": s,
        "relation_uri": p,
        "tail_uri": o if not is_literal else "",
        "is_literal": is_literal
    })

# 提取文字属性三元组，作为实体的特征 (保留这部分以保持兼容性)
entity_features = defaultdict(list)
for s_id, p_name, o_id, s_name, _, o_name, s, p, o, is_literal in filtered_triples:
    if is_literal:
        entity_features[s_id].append({
            "property": p_name,
            "value": str(o)
        })

# 保存实体特征
with open(os.path.join(output_dir, "entity_features.json"), "w", encoding="utf-8") as f:
    json.dump(entity_features, f, indent=2, ensure_ascii=False)

# 第一步：找出所有以"disease"开头的关系
disease_relations = []
for rel_name, _ in relations:
    if rel_name.lower().startswith("disease"):
        disease_relations.append(rel_name)

print(f"找到 {len(disease_relations)} 个以disease开头的关系: {disease_relations}")

# 第二步：找出所有与疾病直接相关的三元组和实体
disease_related_entities = set()
disease_related_triples = []

# 首先添加所有与disease关系直接相关的三元组
for pair in knowledge_pairs:
    if pair["relation"] in disease_relations:
        disease_related_triples.append(pair)
        disease_related_entities.add(pair["head_original"])
        if not pair["is_literal"]:
            disease_related_entities.add(pair["tail_original"])

print(f"初始疾病相关实体数量: {len(disease_related_entities)}")
print(f"初始疾病相关三元组数量: {len(disease_related_triples)}")

# 第三步：迭代扩展相关实体和三元组
# 设置最大迭代次数，防止无限循环
max_iterations = 5
iteration = 0
new_entities_found = True

while new_entities_found and iteration < max_iterations:
    iteration += 1
    initial_entity_count = len(disease_related_entities)
    initial_triple_count = len(disease_related_triples)

    # 临时存储新发现的实体
    new_entities = set()

    # 查找与已知疾病相关实体相关的所有三元组
    for pair in knowledge_pairs:
        # 跳过已经添加的三元组
        if pair in disease_related_triples:
            continue

        head_id = pair["head_original"]
        tail_id = pair["tail_original"]

        # 如果头部实体已在相关实体集合中
        if head_id in disease_related_entities:
            disease_related_triples.append(pair)
            if not pair["is_literal"]:
                new_entities.add(tail_id)

        # 如果尾部实体已在相关实体集合中且不是文字值
        elif not pair["is_literal"] and tail_id in disease_related_entities:
            disease_related_triples.append(pair)
            new_entities.add(head_id)

    # 更新相关实体集合
    disease_related_entities.update(new_entities)

    # 检查是否找到新实体
    new_entities_found = len(disease_related_entities) > initial_entity_count

    print(f"迭代 {iteration}: 新增 {len(disease_related_entities) - initial_entity_count} 个实体, "
          f"新增 {len(disease_related_triples) - initial_triple_count} 个三元组")

# 去除重复的三元组
unique_disease_triples = []
seen_triples = set()

for triple in disease_related_triples:
    # 创建三元组的唯一标识
    triple_id = (triple["head_original"], triple["relation"], triple["tail_original"])
    if triple_id not in seen_triples:
        seen_triples.add(triple_id)
        unique_disease_triples.append(triple)

print(f"最终疾病相关实体数量: {len(disease_related_entities)}")
print(f"最终疾病相关三元组数量: {len(unique_disease_triples)}")

# 保存疾病相关三元组
pd.DataFrame(unique_disease_triples).to_csv(os.path.join(output_dir, "disease_knowledge_graph.csv"), index=False)

# 创建文本版本
with open(os.path.join(output_dir, "disease_knowledge_graph.txt"), "w", encoding="utf-8") as f:
    for pair in unique_disease_triples:
        f.write(f"{pair['head']}\t{pair['relation']}\t{pair['tail']}\n")

# 保存疾病相关实体列表
with open(os.path.join(output_dir, "disease_related_entities.txt"), "w", encoding="utf-8") as f:
    for entity_id in disease_related_entities:
        entity_name = next((name for eid, name, _ in entities if eid == entity_id), entity_id)
        f.write(f"{entity_id}\t{entity_name}\n")
