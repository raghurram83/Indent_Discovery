from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from openai import OpenAI

from .cache import FileCache
from .models import Ask


SYSTEM_PROMPT = "Normalize customer asks into short intent sentences and tags."
USER_TEMPLATE = (
    "Return ONLY JSON: {{ normalized_intent, tags }}.\n"
    "normalized_intent <= 12 words.\n"
    "Choose 1-3 tags from allowed list.\n"
    "Ask: {ask}"
)

ALLOWED_TAGS = [
    "pricing",
    "billing",
    "setup",
    "integrations",
    "renewal",
    "compliance",
    "support",
    "account",
    "ivr",
    "whatsapp",
    "other",
]


class AskNormalizationError(RuntimeError):
    pass


def _call_chat(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _with_retries(fn, retries: int = 3, base_sleep: float = 1.0):
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(base_sleep * (2**attempt))
    if last_err:
        raise last_err


def _parse_json(text: str) -> Dict[str, Any]:
    return json.loads(text)


def normalize_asks(
    atoms_rows: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    cache: FileCache,
    progress_cb=None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = sum(len(a.get("customer_questions", []) or []) for a in atoms_rows)
    done = 0
    for atoms in atoms_rows:
        convo_id = atoms.get("conversation_id", "unknown")
        questions = atoms.get("customer_questions", []) or []
        for idx, question in enumerate(questions):
            if not question:
                continue
            ask_id = f"{convo_id}::q{idx}"
            prompt = USER_TEMPLATE.format(ask=question)
            key = cache.key_for(model, SYSTEM_PROMPT, prompt)
            cached = cache.get(key)
            if cached is None:
                try:
                    raw = _with_retries(lambda: _call_chat(client, model, SYSTEM_PROMPT, prompt))
                    parsed = _parse_json(raw)
                    cache.set(key, parsed)
                    cached = parsed
                except Exception:  # noqa: BLE001
                    cached = {"normalized_intent": question[:120], "tags": ["other"]}

            normalized_intent = str(cached.get("normalized_intent", question)).strip()
            tags = [t for t in (cached.get("tags") or []) if t in ALLOWED_TAGS]
            if not tags:
                tags = ["other"]
            ask = Ask(
                ask_id=ask_id,
                conversation_id=convo_id,
                raw_ask=question,
                normalized_intent=normalized_intent,
                tags=tags,
            )
            results.append(ask.model_dump())
            done += 1
            if progress_cb:
                progress_cb(done, total)

    return results
