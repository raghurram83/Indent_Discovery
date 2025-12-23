from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np
from openai import OpenAI

from .cache import FileCache


def _hash_embedding(text: str, dim: int = 256) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [b / 255.0 for b in digest]
    reps = (dim // len(values)) + 1
    vec = (values * reps)[:dim]
    return vec


def embed_texts(
    texts: List[str],
    client: OpenAI | None,
    model: str,
    cache: FileCache,
) -> Tuple[np.ndarray, List[str]]:
    embeddings = []
    cache_keys = []
    for text in texts:
        key = cache.key_for(model, "embedding", text)
        cached = cache.get(key)
        if cached is None:
            if client is None:
                vec = _hash_embedding(text)
                cache.set(key, vec)
                cached = vec
            else:
                resp = client.embeddings.create(model=model, input=text)
                vec = resp.data[0].embedding
                cache.set(key, vec)
                cached = vec
        embeddings.append(cached)
        cache_keys.append(key)
    # Normalize embedding lengths to avoid ragged arrays from mixed cache/model dims.
    cleaned = []
    max_len = 0
    for vec in embeddings:
        if not isinstance(vec, list):
            vec = list(vec) if vec is not None else []
        if vec:
            max_len = max(max_len, len(vec))
        cleaned.append(vec)
    if max_len == 0:
        return np.zeros((0, 0), dtype=float), cache_keys
    normalized = []
    for vec in cleaned:
        if len(vec) < max_len:
            vec = vec + [0.0] * (max_len - len(vec))
        elif len(vec) > max_len:
            vec = vec[:max_len]
        normalized.append(vec)
    return np.array(normalized, dtype=float), cache_keys
