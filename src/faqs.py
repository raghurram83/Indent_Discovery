from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List

from openai import OpenAI

from .embeddings import embed_texts
from .models import FAQEntry
from .clustering import cluster_embeddings, split_large_clusters


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _dedupe_candidates(candidates: List[str], threshold: float = 0.9) -> List[str]:
    deduped: List[str] = []
    for cand in candidates:
        if not cand:
            continue
        if any(_similar(cand, existing) >= threshold for existing in deduped):
            continue
        deduped.append(cand)
    return deduped


def _collect_answer_candidates(conversation: Dict[str, Any], question: str) -> List[str]:
    if not conversation:
        return []
    messages = conversation.get("messages", [])
    question_lower = question.lower()
    answers: List[str] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "customer":
            continue
        text = msg.get("text", "")
        if not text:
            continue
        is_match = question_lower in text.lower() or text.strip().endswith("?")
        if not is_match:
            continue
        for offset in (1, 2):
            if idx + offset >= len(messages):
                continue
            nxt = messages[idx + offset]
            if nxt.get("role") == "agent" and nxt.get("text"):
                answers.append(nxt.get("text"))
    return answers


def build_faq_catalogue(
    atoms_rows: List[Dict[str, Any]],
    conversations: List[Dict[str, Any]],
    client: OpenAI | None,
    embed_model: str,
    cache,
    threshold: float = 0.82,
    intent_catalogue: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    questions = []
    question_meta = []
    convo_map = {c.get("conversation_id"): c for c in conversations}

    for atoms in atoms_rows:
        convo_id = atoms.get("conversation_id")
        for q in atoms.get("customer_questions", []) or []:
            if not q:
                continue
            questions.append(q)
            question_meta.append({"conversation_id": convo_id, "question": q})

    if not questions:
        return []

    embeddings, _ = embed_texts(questions, client, embed_model, cache)
    clusters = split_large_clusters(cluster_embeddings(embeddings, threshold), embeddings)

    faq_entries: List[FAQEntry] = []
    for idx, cluster in enumerate(clusters):
        cluster_questions = [questions[i] for i in cluster]
        source_conversations = sorted(
            {question_meta[i]["conversation_id"] for i in cluster if question_meta[i]["conversation_id"]}
        )
        canonical = min(cluster_questions, key=len)
        variants = cluster_questions[:10]

        answer_candidates: List[str] = []
        for meta in (question_meta[i] for i in cluster):
            convo = convo_map.get(meta["conversation_id"])
            answer_candidates.extend(_collect_answer_candidates(convo, meta["question"]))
        answer_candidates = _dedupe_candidates(answer_candidates)

        linked_intent = None
        if intent_catalogue:
            question_tokens = set(canonical.lower().split())
            for intent in intent_catalogue.get("parent_intents", []):
                intent_name = intent.get("intent_name", "")
                if not intent_name:
                    continue
                intent_tokens = set(intent_name.lower().split())
                if question_tokens.intersection(intent_tokens):
                    linked_intent = intent_name
                    break

        faq = FAQEntry(
            faq_id=f"faq_{idx}",
            canonical_question=canonical,
            variants=variants,
            answer_candidates=answer_candidates,
            status="needs_verification",
            source_conversations=source_conversations,
            linked_intent=linked_intent,
        )
        faq_entries.append(faq)

    return [f.model_dump() for f in faq_entries]
