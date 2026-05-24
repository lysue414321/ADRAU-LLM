#!/usr/bin/env python3
"""
ADRAU-LLM Antibiotic QA Pair Generator
======================================
Generates question-answer pairs from antibiotic clinical guidelines via a
two-stage pipeline: (1) knowledge graph extraction and template-based QA
generation, followed by (2) LLM-based dimension-aware generation with
hallucination validation.

Pipeline Overview:
    1. Guideline text  -->  Knowledge Graph (entity-relation-entity triples)
    2. KG triples       -->  Template-based QA pairs (forward + reverse)
    3. Guideline text  -->  LLM-generated dimension-specific QA pairs
    4. All QA pairs     -->  Hallucination check via source-text grounding

Usage:
    python antibiotic_qa_generator.py \
        --guidelines data/guidelines.jsonl \
        --kg data/knowledge_graph.json \
        --output data/qa_pairs.json \
        --temperature 0.3 --top_p 0.9 --max_qa_per_dim 500
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qa_generator")


# ===================================================================
# Data Structures
# ===================================================================

@dataclass
class Triple:
    """A single knowledge graph triple."""

    subject: str
    relation: str
    object: str
    source: str  # raw guideline passage
    confidence: float = 1.0


@dataclass
class QAPair:
    """A generated question-answer pair."""

    question: str
    answer: str
    category: str  # disease_treatment, pathogen_matching, special_populations, contraindications
    guideline_source: str
    generation_method: str  # "kg_template" or "llm_generation"
    hallucination_score: float = 0.0
    validated: bool = False


# ===================================================================
# Template Registry for KG-based Generation
# ===================================================================

TEMPLATES_FORWARD = {
    "first_line_treatment": "对于{subject}，首选的抗菌药物治疗方案是什么？",
    "alternative_treatment": "对于{subject}的替代抗菌药物治疗方案有哪些？",
    "caused_by": "{subject}常见由哪些病原体引起？",
    "contraindicated_in": "哪些人群不宜使用{subject}？",
    "dose_adjustment": "在{object}情况下，{subject}的剂量应如何调整？",
    "adverse_effect": "{subject}的主要不良反应有哪些，应如何监测？",
    "drug_interaction": "{subject}与哪些药物存在相互作用，如何管理？",
    "resistance_pattern": "{subject}的常见耐药机制是什么，应如何选择替代治疗？",
    "duration": "{subject}的推荐抗菌药物疗程是多长时间？",
    "monitoring": "使用{subject}进行治疗时需要监测哪些临床指标？",
}

TEMPLATES_REVERSE = {
    "first_line_treatment": "当需要使用{object}作为一线治疗时，可能针对的疾病是什么？",
    "caused_by": "{object}常见可引起哪些感染性疾病？",
    "contraindicated_in": "在{object}情况下禁用的抗菌药物有哪些？",
    "treats": "{object}可用于治疗哪些感染性疾病？",
}


# ===================================================================
# LLM API Client (mock / pseudocode)
# ===================================================================

class LLMClient:
    """Thin wrapper around the Qwen-Plus API for QA pair generation.

    In production this would use the official DashScope / Alibaba Cloud SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the LLM client.

        Parameters
        ----------
        api_key : str, optional
            API key. Defaults to ``DASHSCOPE_API_KEY`` env var.
        model : str
            Model identifier.
        temperature : float
            Sampling temperature (0.0-2.0).
        top_p : float
            Nucleus sampling threshold.
        max_tokens : int
            Maximum tokens per completion.
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call the LLM API and return the generated text.

        This is pseudocode representing a production integration.
        In practice, this would use ``dashscope.Generation.call(...)``.

        Parameters
        ----------
        system_prompt : str
            System-level instruction.
        user_prompt : str
            User message with the task.

        Returns
        -------
        str
            Model-generated text.
        """
        # PSEUDOCODE -- replace with actual API call:
        # import dashscope
        # response = dashscope.Generation.call(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     temperature=self.temperature,
        #     top_p=self.top_p,
        #     max_tokens=self.max_tokens,
        # )
        # return response.output.choices[0].message.content

        logger.debug(
            "LLM call: model=%s, temp=%.2f, top_p=%.2f, prompt_len=%d",
            self.model,
            self.temperature,
            self.top_p,
            len(user_prompt),
        )
        # Simulated response (placeholder)
        return json.dumps({
            "question": "[PLACEHOLDER] Generated question text",
            "answer": "[PLACEHOLDER] Generated answer text",
        }, ensure_ascii=False)


# ===================================================================
# KG-to-QA Generation
# ===================================================================

def kg_to_qa_pairs(
    triples: list[Triple],
    templates_forward: Optional[dict[str, str]] = None,
    templates_reverse: Optional[dict[str, str]] = None,
) -> list[QAPair]:
    """Generate QA pairs from knowledge graph triples using template-based
    forward and reverse question construction.

    Forward questions use the subject as the query focus (e.g., "What is the
    first-line treatment for X?"). Reverse questions invert the triple
    (e.g., "Which diseases are treated by Y?").

    Parameters
    ----------
    triples : list[Triple]
        Knowledge graph triples extracted from guidelines.
    templates_forward : dict, optional
        Mapping of relation_name -> question template (forward).
    templates_reverse : dict, optional
        Mapping of relation_name -> question template (reverse).

    Returns
    -------
    list[QAPair]
        Generated QA pairs.
    """
    if templates_forward is None:
        templates_forward = TEMPLATES_FORWARD
    if templates_reverse is None:
        templates_reverse = TEMPLATES_REVERSE

    qa_pairs: list[QAPair] = []
    logger.info("Generating QA pairs from %d KG triples.", len(triples))

    for triple in triples:
        relation = triple.relation

        # --- Forward generation ---
        if relation in templates_forward:
            question = templates_forward[relation].format(
                subject=triple.subject,
                object=triple.object,
            )
            answer = (
                f"根据{t['guideline_source']}，{triple.subject}的{relation}为{triple.object}。"
            )
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                category=_infer_category(relation),
                guideline_source=triple.source,
                generation_method="kg_template_forward",
            ))

        # --- Reverse generation ---
        if relation in templates_reverse:
            question = templates_reverse[relation].format(
                subject=triple.subject,
                object=triple.object,
            )
            answer = (
                f"根据{t['guideline_source']}，{triple.object}可用于治疗{triple.subject}。"
            )
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                category=_infer_category(relation),
                guideline_source=triple.source,
                generation_method="kg_template_reverse",
            ))

    logger.info("KG-based generation produced %d QA pairs.", len(qa_pairs))
    return qa_pairs


def _infer_category(relation: str) -> str:
    """Map a KG relation to a QA category label.

    Parameters
    ----------
    relation : str
        Knowledge graph relation name.

    Returns
    -------
    str
        Category: disease_treatment, pathogen_matching, special_populations,
        or contraindications.
    """
    category_map = {
        "first_line_treatment": "disease_treatment",
        "alternative_treatment": "disease_treatment",
        "duration": "disease_treatment",
        "monitoring": "disease_treatment",
        "caused_by": "pathogen_matching",
        "resistance_pattern": "pathogen_matching",
        "treats": "pathogen_matching",
        "contraindicated_in": "contraindications",
        "adverse_effect": "contraindications",
        "drug_interaction": "contraindications",
        "dose_adjustment": "special_populations",
    }
    return category_map.get(relation, "disease_treatment")


# ===================================================================
# Guideline-to-QA Generation (LLM-based)
# ===================================================================

DIMENSION_PROMPTS = {
    "disease_treatment": (
        "你是一位临床药学专家。请根据以下抗菌药物指南段落，生成一个关于"
        "「疾病治疗方案选择」的中文问答对。问题应聚焦于：针对某疾病，"
        "如何根据病情严重程度、病原体可能性和患者特征经验性选择抗菌药物方案。"
        "答案应引用指南原文的依据，使用中国的药物通用名，包含剂量和疗程信息。"
        "输出格式为JSON：{{\"question\": \"...\", \"answer\": \"...\"}}"
    ),
    "pathogen_matching": (
        "你是一位微生物学与感染病学专家。请根据以下抗菌药物指南段落，生成一个关于"
        "「病原体-药物匹配」的中文问答对。问题应聚焦于：特定病原体感染应选择"
        "何种抗菌药物，替代方案有哪些，以及如何根据耐药性调整治疗。"
        "答案需明确列出首选和替代药物，附上指南依据。"
        "输出格式为JSON：{{\"question\": \"...\", \"answer\": \"...\"}}"
    ),
    "special_populations": (
        "你是一位临床药学专家。请根据以下抗菌药物指南段落，生成一个关于"
        "「特殊人群用药调整」的中文问答对。问题应聚焦于："
        "儿童、老年人、孕妇、哺乳期妇女、肝肾功能不全患者等特殊人群的"
        "抗菌药物选择及剂量调整方案。"
        "答案应包含Cockcroft-Gault或eGFR计算的参考，以及具体的减量方案。"
        "输出格式为JSON：{{\"question\": \"...\", \"answer\": \"...\"}}"
    ),
    "contraindications": (
        "你是一位药物警戒与临床安全专家。请根据以下抗菌药物指南段落，生成一个关于"
        "「抗菌药物禁忌证及安全用药」的中文问答对。问题应聚焦于："
        "某种或某类抗菌药物的禁忌证、慎用情况、交叉过敏反应、不良反应监测要点。"
        "答案应详细列出绝对禁忌证、相对禁忌证和相关的安全监测指标。"
        "输出格式为JSON：{{\"question\": \"...\", \"answer\": \"...\"}}"
    ),
}


def guideline_to_qa_pairs(
    guideline_passages: list[dict[str, str]],
    llm_client: LLMClient,
    max_pairs_per_dimension: int = 500,
) -> list[QAPair]:
    """Generate QA pairs from guideline text passages using an LLM.

    For each dimension (disease_treatment, pathogen_matching, special_populations,
    contraindications), the function samples passages, constructs a dimension-
    specific system prompt, calls the LLM to generate a question, then calls
    the LLM again to generate the corresponding answer.

    Parameters
    ----------
    guideline_passages : list[dict]
        List of dicts with keys ``text`` (the guideline paragraph) and
        ``source`` (the guideline name).
    llm_client : LLMClient
        Configured LLM client for generation.
    max_pairs_per_dimension : int
        Maximum number of QA pairs to generate per dimension.

    Returns
    -------
    list[QAPair]
        Generated QA pairs.
    """
    qa_pairs: list[QAPair] = []

    for dimension, system_prompt in DIMENSION_PROMPTS.items():
        logger.info(
            "Generating LLM-based QA pairs for dimension: %s", dimension
        )

        # Shuffle passages for diversity and take a sample
        rng = np.random.default_rng(42)
        passages_sample = rng.choice(
            guideline_passages,
            size=min(max_pairs_per_dimension, len(guideline_passages)),
            replace=False,
        )

        for i, passage in enumerate(passages_sample):
            if i >= max_pairs_per_dimension:
                break

            # Step 1: Generate question
            question_prompt = (
                f"{system_prompt}\n\n"
                f"请根据以下指南段落生成一个问答对：\n\n"
                f"【指南段落】\n{passage['text']}\n\n"
                f"【指南来源】{passage['source']}"
            )

            try:
                raw_response = llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=question_prompt,
                )
                parsed = json.loads(raw_response)

                qa_pairs.append(QAPair(
                    question=parsed.get("question", ""),
                    answer=parsed.get("answer", ""),
                    category=dimension,
                    guideline_source=passage["source"],
                    generation_method="llm_generation",
                ))

                if (i + 1) % 50 == 0:
                    logger.info(
                        "  [%s] Generated %d / %d pairs.",
                        dimension,
                        i + 1,
                        min(max_pairs_per_dimension, len(passages_sample)),
                    )

                # Rate limiting
                time.sleep(0.5)

            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse LLM output for passage %d in dimension %s.",
                    i,
                    dimension,
                )
                continue
            except Exception as exc:
                logger.error("LLM generation error: %s", exc)
                continue

        logger.info(
            "Dimension '%s' complete: %d QA pairs generated.",
            dimension,
            len([p for p in qa_pairs if p.category == dimension]),
        )

    return qa_pairs


# ===================================================================
# Hallucination Validation
# ===================================================================

def validate_qa_pairs(
    qa_pairs: list[QAPair],
    source_texts: list[str],
    llm_client: Optional[LLMClient] = None,
    threshold: float = 0.5,
) -> list[QAPair]:
    """Validate QA pairs for hallucination by checking factual grounding
    against source guideline texts.

    Uses a two-pronged approach:
    1. **N-gram overlap**: Compute the proportion of answer n-grams (2-, 3-,
       and 4-grams) that appear in any source passage.
    2. **LLM-based verification** (optional, expensive): Ask the LLM to judge
       whether each sentence in the answer is supported by the source texts.

    Pairs with a hallucination score > ``threshold`` are marked
    ``validated = True``. The hallucination score ranges from 0 (fully
    hallucinated) to 1 (fully grounded in source texts).

    Parameters
    ----------
    qa_pairs : list[QAPair]
        QA pairs to validate.
    source_texts : list[str]
        The full set of guideline text passages used as ground truth.
    llm_client : LLMClient, optional
        If provided, use LLM-based verification in addition to n-gram overlap.
    threshold : float
        Minimum hallucination score to mark as validated.

    Returns
    -------
    list[QAPair]
        QA pairs with ``hallucination_score`` and ``validated`` fields populated.
    """
    logger.info(
        "Validating %d QA pairs against %d source passages.",
        len(qa_pairs),
        len(source_texts),
    )

    # Precompute source n-gram sets for efficiency
    all_source_text = " ".join(source_texts)
    source_ngrams_2 = _ngrams(all_source_text, 2)
    source_ngrams_3 = _ngrams(all_source_text, 3)
    source_ngrams_4 = _ngrams(all_source_text, 4)

    for pair in qa_pairs:
        # --- N-gram overlap score ---
        answer_text = pair.answer
        answer_ngrams_2 = _ngrams(answer_text, 2)
        answer_ngrams_3 = _ngrams(answer_text, 3)
        answer_ngrams_4 = _ngrams(answer_text, 4)

        overlap_2 = _overlap_ratio(answer_ngrams_2, source_ngrams_2)
        overlap_3 = _overlap_ratio(answer_ngrams_3, source_ngrams_3)
        overlap_4 = _overlap_ratio(answer_ngrams_4, source_ngrams_4)

        # Weighted combination (longer n-grams count more)
        ngram_score = (0.2 * overlap_2) + (0.3 * overlap_3) + (0.5 * overlap_4)

        # --- LLM-based verification (optional) ---
        llm_score = 1.0  # default if no LLM verification
        if llm_client is not None:
            llm_score = _llm_verify(pair, source_texts, llm_client)

        # Combined score
        pair.hallucination_score = 0.5 * ngram_score + 0.5 * llm_score
        pair.validated = pair.hallucination_score >= threshold

    n_validated = sum(1 for p in qa_pairs if p.validated)
    n_rejected = len(qa_pairs) - n_validated
    logger.info(
        "Validation complete: %d validated, %d rejected (threshold=%.2f).",
        n_validated,
        n_rejected,
        threshold,
    )
    return qa_pairs


def _ngrams(text: str, n: int) -> set[str]:
    """Extract character-level n-grams from text.

    Parameters
    ----------
    text : str
        Input text.
    n : int
        N-gram size (2, 3, or 4).

    Returns
    -------
    set[str]
        Set of n-grams.
    """
    cleaned = re.sub(r"\s+", "", text)
    return {cleaned[i:i + n] for i in range(max(0, len(cleaned) - n + 1))}


def _overlap_ratio(
    answer_ngrams: set[str],
    source_ngrams: set[str],
) -> float:
    """Compute Jaccard-like overlap ratio: |intersection| / |answer_ngrams|.

    Parameters
    ----------
    answer_ngrams : set[str]
        N-grams from the answer.
    source_ngrams : set[str]
        N-grams from all source texts.

    Returns
    -------
    float
        Overlap ratio in [0, 1].
    """
    if not answer_ngrams:
        return 0.0
    intersection = answer_ngrams & source_ngrams
    return len(intersection) / len(answer_ngrams)


def _llm_verify(
    pair: QAPair,
    source_texts: list[str],
    llm_client: LLMClient,
) -> float:
    """Use the LLM to verify whether the answer is supported by source texts.

    This is a costly verification method and should be used sparingly.

    Parameters
    ----------
    pair : QAPair
        The QA pair to verify.
    source_texts : list[str]
        Source guideline passages.
    llm_client : LLMClient
        LLM client for verification.

    Returns
    -------
    float
        LLM-assigned factuality score in [0, 1].
    """
    system_prompt = (
        "你是一个严格的医学内容审核专家。请判断以下答案中的信息是否全部"
        "可以在提供的参考文献段落中找到支持。逐句检查。\n"
        "输出JSON格式：{\"supported_sentences\": N, \"total_sentences\": M, "
        "\"issues\": [\"...\"], \"score\": 0.0-1.0}"
    )

    source_joined = "\n\n---\n\n".join(source_texts[:5])  # first 5 for context
    user_prompt = (
        f"问题：{pair.question}\n\n"
        f"答案：{pair.answer}\n\n"
        f"参考来源：\n{source_joined}"
    )

    try:
        response = llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        result = json.loads(response)
        return float(result.get("score", 0.5))
    except Exception:
        logger.warning("LLM verification failed; defaulting to score=0.5")
        return 0.5


# ===================================================================
# Orchestration
# ===================================================================

def main() -> None:
    """Run the full QA pair generation pipeline."""
    parser = argparse.ArgumentParser(
        description="ADRAU-LLM: Generate antibiotic QA pairs from guidelines."
    )
    parser.add_argument(
        "--guidelines",
        required=True,
        help="Path to guidelines JSONL file "
             "({text: ..., source: ...} per line).",
    )
    parser.add_argument(
        "--kg",
        default=None,
        help="Path to pre-extracted knowledge graph JSON (list of triples).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSON file for QA pairs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature (default: 0.3).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="LLM nucleus sampling top_p (default: 0.9).",
    )
    parser.add_argument(
        "--max_qa_per_dim",
        type=int,
        default=500,
        help="Max QA pairs per dimension (default: 500).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run hallucination validation after generation.",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.5,
        help="Minimum hallucination score to accept a pair (default: 0.5).",
    )
    parser.add_argument(
        "--llm-verify",
        action="store_true",
        help="Use LLM-based hallucination verification (expensive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load guidelines
    # ------------------------------------------------------------------
    logger.info("Loading guidelines from %s", args.guidelines)
    guideline_passages: list[dict[str, str]] = []
    with open(args.guidelines, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                guideline_passages.append(json.loads(line))
    logger.info("Loaded %d guideline passages.", len(guideline_passages))

    all_source_texts = [p["text"] for p in guideline_passages]

    # ------------------------------------------------------------------
    # Initialize LLM client
    # ------------------------------------------------------------------
    llm_client = LLMClient(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # ------------------------------------------------------------------
    # Stage 1: KG-based generation
    # ------------------------------------------------------------------
    kg_qa_pairs: list[QAPair] = []
    if args.kg:
        logger.info("Loading knowledge graph from %s", args.kg)
        with open(args.kg, "r", encoding="utf-8") as fh:
            kg_data = json.load(fh)

        triples = [
            Triple(
                subject=t["subject"],
                relation=t["relation"],
                object=t["object"],
                source=t.get("source", ""),
                confidence=t.get("confidence", 1.0),
            )
            for t in kg_data
        ]
        kg_qa_pairs = kg_to_qa_pairs(triples)
    else:
        logger.info(
            "No KG file provided; skipping KG-based QA generation."
        )

    # ------------------------------------------------------------------
    # Stage 2: LLM-based generation
    # ------------------------------------------------------------------
    llm_qa_pairs = guideline_to_qa_pairs(
        guideline_passages=guideline_passages,
        llm_client=llm_client,
        max_pairs_per_dimension=args.max_qa_per_dim,
    )

    # ------------------------------------------------------------------
    # Combine and deduplicate
    # ------------------------------------------------------------------
    all_pairs = kg_qa_pairs + llm_qa_pairs
    logger.info(
        "Total generated: %d (KG: %d, LLM: %d).",
        len(all_pairs),
        len(kg_qa_pairs),
        len(llm_qa_pairs),
    )

    # Deduplicate by question hash
    seen: set[str] = set()
    unique_pairs: list[QAPair] = []
    for pair in all_pairs:
        q_hash = hashlib.md5(pair.question.encode("utf-8")).hexdigest()
        if q_hash not in seen:
            seen.add(q_hash)
            unique_pairs.append(pair)

    logger.info(
        "After deduplication: %d unique QA pairs (removed %d duplicates).",
        len(unique_pairs),
        len(all_pairs) - len(unique_pairs),
    )

    # ------------------------------------------------------------------
    # Stage 3: Validation (optional)
    # ------------------------------------------------------------------
    if args.validate:
        verify_client = llm_client if args.llm_verify else None
        unique_pairs = validate_qa_pairs(
            qa_pairs=unique_pairs,
            source_texts=all_source_texts,
            llm_client=verify_client,
            threshold=args.validation_threshold,
        )

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "metadata": {
            "total_pairs": len(unique_pairs),
            "validated_pairs": sum(1 for p in unique_pairs if p.validated),
            "generation_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_qa_per_dim": args.max_qa_per_dim,
            },
        },
        "qa_pairs": [
            {
                "question": p.question,
                "answer": p.answer,
                "category": p.category,
                "guideline_source": p.guideline_source,
                "generation_method": p.generation_method,
                "hallucination_score": round(p.hallucination_score, 4),
                "validated": p.validated,
            }
            for p in unique_pairs
        ],
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, ensure_ascii=False, indent=2)

    logger.info(
        "QA pairs saved to %s (%d pairs total, %d validated).",
        args.output,
        len(unique_pairs),
        sum(1 for p in unique_pairs if p.validated),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    category_counts = {}
    for p in unique_pairs:
        category_counts[p.category] = category_counts.get(p.category, 0) + 1

    logger.info("QA pair distribution by category:")
    for cat, count in sorted(category_counts.items()):
        logger.info("  %s: %d", cat, count)


if __name__ == "__main__":
    main()
