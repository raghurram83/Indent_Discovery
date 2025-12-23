from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict

from openai import OpenAI

from .cache import FileCache


def call_chat(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def with_retries(fn: Callable[[], Any], retries: int = 3, base_sleep: float = 1.0) -> Any:
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(base_sleep * (2**attempt))
    if last_err:
        raise last_err
    return None


def parse_json(text: str) -> Dict[str, Any]:
    return json.loads(text)


def repair_json(
    client: OpenAI,
    model: str,
    broken_json: str,
    system: str = "Fix JSON formatting only.",
    user_template: str = "Return valid JSON only. Do not change meaning.\n{broken_json}",
) -> Dict[str, Any]:
    user = user_template.format(broken_json=broken_json)
    response = with_retries(lambda: call_chat(client, model, system, user))
    return parse_json(response)


def chat_json_with_cache(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    cache: FileCache,
    cache_key: str,
    repair_system: str = "Fix JSON formatting only.",
    repair_user: str = "Return valid JSON only. Do not change meaning.\n{broken_json}",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    raw = with_retries(lambda: call_chat(client, model, system, user, temperature=temperature))
    try:
        parsed = parse_json(raw)
    except Exception:  # noqa: BLE001
        parsed = repair_json(client, model, raw, system=repair_system, user_template=repair_user)
    cache.set(cache_key, parsed)
    return parsed
