from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Dict, List

from openai import OpenAI

from .cache import FileCache
from .models import IntentCatalogue, ParentIntent


SYSTEM_PROMPT = "Consolidate similar asks into parent intents and scenarios. Split if mixed."
USER_TEMPLATE = (
    "Given these asks, output ONLY JSON:\n"
    "{{\n"
    "  'parent_intents': [\n"
    "    {{\n"
    "      'intent_name': string,\n"
    "      'definition': string,\n"
    "      'scenarios': [\n"
    "        {{ 'scenario_name': string, 'example_asks': string[] }}\n"
    "      ]\n"
    "    }}\n"
    "  ]\n"
    "}}\n"
    "Rules:\n"
    "- Split into multiple parent_intents if mixed topics.\n"
    "- Keep intent_name stable and short.\n\n"
    "Asks:\n{asks}"
)


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


def build_intent_catalogue(
    asks: List[Dict[str, Any]],
    clusters: List[List[int]],
    client: OpenAI,
    model: str,
    cache: FileCache,
    min_freq: int = 3,
) -> Dict[str, Any]:
    intents: List[ParentIntent] = []
    edge_cases: List[Dict[str, Any]] = []

    special_tags = {"billing", "compliance", "account", "renewal"}

    for cluster_id, cluster in enumerate(clusters):
        cluster_asks = [asks[idx] for idx in cluster]
        sample = cluster_asks[:30]
        asks_text = json.dumps([a["normalized_intent"] for a in sample], ensure_ascii=True)
        prompt = USER_TEMPLATE.format(asks=asks_text)
        key = cache.key_for(model, SYSTEM_PROMPT, prompt)
        cached = cache.get(key)
        if cached is None:
            raw = _with_retries(lambda: _call_chat(client, model, SYSTEM_PROMPT, prompt))
            try:
                parsed = json.loads(raw)
            except Exception:  # noqa: BLE001
                parsed = {"parent_intents": []}
            cache.set(key, parsed)
            cached = parsed

        parent_intents = cached.get("parent_intents", []) if isinstance(cached, dict) else []
        cluster_tags = Counter(tag for a in cluster_asks for tag in a.get("tags", []))

        for parent in parent_intents:
            try:
                intent = ParentIntent(**parent)
            except Exception:  # noqa: BLE001
                continue
            freq = len(cluster_asks)
            has_special = any(tag in special_tags for tag in cluster_tags.keys())
            if freq >= min_freq or has_special:
                intents.append(intent)
            else:
                edge_cases.append(
                    {
                        "cluster_id": cluster_id,
                        "intent_name": intent.intent_name,
                        "reason": "below_min_freq",
                        "sample_asks": [a["normalized_intent"] for a in sample],
                    }
                )

    catalogue = IntentCatalogue(parent_intents=intents, edge_cases=edge_cases)
    return catalogue.model_dump()
