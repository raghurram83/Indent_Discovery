from __future__ import annotations

from typing import Dict, List

import numpy as np


def _cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def cluster_embeddings(embeddings: np.ndarray, threshold: float = 0.82) -> List[List[int]]:
    if len(embeddings) == 0:
        return []
    sims = _cosine_similarity(embeddings)
    uf = UnionFind(len(embeddings))
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if sims[i, j] >= threshold:
                uf.union(i, j)
    clusters: Dict[int, List[int]] = {}
    for idx in range(len(embeddings)):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def split_large_clusters(clusters: List[List[int]], embeddings: np.ndarray, max_size: int = 60) -> List[List[int]]:
    if not clusters:
        return []
    final_clusters: List[List[int]] = []
    for cluster in clusters:
        if len(cluster) <= max_size:
            final_clusters.append(cluster)
            continue
        for idx in range(0, len(cluster), max_size):
            final_clusters.append(cluster[idx : idx + max_size])
    return final_clusters
