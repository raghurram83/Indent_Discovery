from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .cache import FileCache
from .clustering import cluster_embeddings, split_large_clusters
from .embeddings import embed_texts
from .models import FAQEntry, ValueScore
from .openai_client import chat_json_with_cache


HIGH_IMPACT_TERMS = ["billing", "compliance", "renewal"]

SYSTEM_PROMPT = (
    "You consolidate similar FAQ questions and draft a KB-style answer using provided candidates. "
    "Always respond in English."
)
USER_TEMPLATE = (
    "Given:\n"
    "- question variants (q_raw and q_clean)\n"
    "- answer candidates list (grounded)\n"
    "Output ONLY JSON:\n"
    "{{\n"
    "  'faq_id': string,\n"
    "  'canonical_question': string,\n"
    "  'question_variants': string[],\n"
    "  'draft_answer': string,\n"
    "  'needs_verification': boolean,\n"
    "  'value_score': {{ '0_to_5': number, 'reason': string }}\n"
    "}}\n"
    "Rules:\n"
    "- canonical_question must be clear and self-contained\n"
    "- draft_answer must ONLY use provided answer candidates; if candidates are 'NEEDS_INFO', keep draft_answer as what info is needed.\n"
    "- value_score: rate usefulness for future customers based on frequency + generality.\n\n"
    "Question variants:\n{variants}\n\n"
    "Answer candidates:\n{answers}\n"
)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        cleaned = (item or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _cluster_rows(
    faq_rows: List[Dict[str, Any]],
    client: OpenAI | None,
    embed_model: str,
    embed_cache: FileCache,
    threshold: float,
) -> List[List[int]]:
    questions = [row["q_clean"] for row in faq_rows]
    embeddings, _ = embed_texts(questions, client, embed_model, embed_cache)
    return split_large_clusters(cluster_embeddings(embeddings, threshold), embeddings)


def _has_high_impact(questions: List[str]) -> bool:
    lowered = " ".join(questions).lower()
    return any(term in lowered for term in HIGH_IMPACT_TERMS)


def _fallback_value_score(freq: int, reason: str) -> ValueScore:
    score = 1.0 if freq <= 1 else 2.5 if freq <= 3 else 3.5
    return ValueScore.model_validate({"0_to_5": score, "reason": reason})


def _fallback_faq(faq_id: str, variants: List[str], answers: List[str], freq: int) -> Dict[str, Any]:
    canonical = min(variants, key=len) if variants else "Unknown question"
    if answers:
        draft = answers[0]
        needs_verification = any(a.startswith("NEEDS_INFO:") for a in answers)
        reason = "heuristic_from_candidates"
    else:
        draft = "NEEDS_INFO: missing answer details from agent"
        needs_verification = True
        reason = "no_answer_candidates"
    value_score = _fallback_value_score(freq, reason)
    entry = FAQEntry(
        faq_id=faq_id,
        canonical_question=canonical,
        question_variants=variants,
        draft_answer=draft,
        needs_verification=needs_verification,
        value_score=value_score,
    )
    return entry.model_dump(by_alias=True)


def build_faq_catalogue(
    atoms_rows: List[Dict[str, Any]],
    client: OpenAI | None,
    llm_model: str,
    llm_cache: FileCache,
    embed_model: str,
    embed_cache: FileCache,
    system_prompt: str = SYSTEM_PROMPT,
    user_template: str = USER_TEMPLATE,
    threshold: float = 0.82,
    min_freq: int = 3,
) -> List[Dict[str, Any]]:
    faq_rows: List[Dict[str, Any]] = []
    for atoms in atoms_rows:
        convo_id = atoms.get("conversation_id")
        for candidate in atoms.get("faq_candidates", []) or []:
            q_clean = (candidate.get("q_clean") or "").strip()
            if not q_clean:
                continue
            faq_rows.append(
                {
                    "conversation_id": convo_id,
                    "q_clean": q_clean,
                    "q_raw": (candidate.get("q_raw") or "").strip(),
                    "a_candidates": candidate.get("a_candidates") or [],
                }
            )

    if not faq_rows:
        return []

    clusters = _cluster_rows(faq_rows, client, embed_model, embed_cache, threshold)
    results: List[Dict[str, Any]] = []
    for idx, cluster in enumerate(clusters):
        cluster_rows = [faq_rows[i] for i in cluster]
        freq = len(cluster_rows)
        questions = [row["q_clean"] for row in cluster_rows]
        variants = _dedupe([row["q_raw"] for row in cluster_rows] + questions)
        answers = _dedupe([ans for row in cluster_rows for ans in (row["a_candidates"] or [])])

        has_impact = _has_high_impact(questions)
        should_llm = client is not None and (freq >= min_freq or has_impact)

        if should_llm:
            variants_json = json.dumps(variants, ensure_ascii=True)
            answers_json = json.dumps(answers, ensure_ascii=True)
            prompt = user_template
            if "{variants}" in prompt:
                prompt = prompt.replace("{variants}", variants_json)
            else:
                prompt = f"{prompt}\n\nQuestion variants:\n{variants_json}"
            if "{answers}" in prompt:
                prompt = prompt.replace("{answers}", answers_json)
            else:
                prompt = f"{prompt}\n\nAnswer candidates:\n{answers_json}"
            key = llm_cache.key_for(llm_model, system_prompt, prompt)
            try:
                parsed = chat_json_with_cache(
                    client,
                    llm_model,
                    system_prompt,
                    prompt,
                    llm_cache,
                    key,
                )
                parsed["faq_id"] = parsed.get("faq_id") or f"faq_{idx}"
                entry = FAQEntry.model_validate(parsed)
                results.append(entry.model_dump(by_alias=True))
                continue
            except Exception:  # noqa: BLE001
                pass

        results.append(_fallback_faq(f"faq_{idx}", variants, answers, freq))

    return results
