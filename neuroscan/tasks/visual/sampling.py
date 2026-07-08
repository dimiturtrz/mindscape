"""Batch construction for the contrastive retrieval trainer — balance + hardness of the InfoNCE negatives.

Uniform-random batching lets frequent concepts dominate and makes the in-batch negatives arbitrary. Two
model-free strategies (both pure — index math + a precomputed CLIP matrix, no EEG model needed, so no
cold-start):

  - `stratified_batches` (bd ewd): round-robin over concepts so each batch spans many distinct concepts
    ~equally — balanced negatives across the label space.
  - `clip_neighbor_groups` (bd mnr): for each concept, its CLIP-nearest OTHER concepts. The retrieval TARGET
    is CLIP, so semantic confusability (husky~wolf) is known up front, model-free — the seed for building
    batches whose negatives are HARD from epoch 0 (no first pass through the encoder needed).
"""
from __future__ import annotations

import numpy as np


def stratified_batches(concept_ids: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Index batches balanced across concepts: shuffle within each concept, then round-robin draw so every
    batch sees many distinct concepts ~equally (balances the InfoNCE negatives vs uniform sampling)."""
    by_concept: dict[int, list[int]] = {}
    for i, concept in enumerate(concept_ids):
        by_concept.setdefault(int(concept), []).append(i)
    pools = [rng.permutation(idx).tolist() for idx in by_concept.values()]
    order: list[int] = []
    while any(pools):
        for pool in pools:
            if pool:
                order.append(pool.pop())
    return [np.asarray(order[i:i + batch_size]) for i in range(0, len(order), batch_size)]


def balanced_batches(concept_ids: np.ndarray, concepts_per_batch: int, samples_per_concept: int,
                     rng: np.random.Generator, n_batches: int | None = None) -> list[np.ndarray]:
    """STRICT balanced batches (bd 2j2): each batch = `concepts_per_batch` concepts × `samples_per_concept`
    trials each, so every concept in the batch is represented EQUALLY (unlike `stratified_batches`' round-robin,
    which decays as scarce concepts empty out). Scarce concepts are drawn WITH replacement to keep the count
    exact. `n_batches` defaults to one pass over the data. Balances the InfoNCE negatives across the label space
    every step."""
    by_concept: dict[int, list[int]] = {}
    for i, concept in enumerate(concept_ids):
        by_concept.setdefault(int(concept), []).append(i)
    concepts = list(by_concept)
    per_batch = min(concepts_per_batch, len(concepts))
    if n_batches is None:
        n_batches = max(1, len(concept_ids) // (per_batch * samples_per_concept))
    batches = []
    for _ in range(n_batches):
        chosen = rng.choice(concepts, size=per_batch, replace=False)
        idx: list[int] = []
        for concept in chosen:
            pool = by_concept[int(concept)]
            idx.extend(rng.choice(pool, size=samples_per_concept, replace=len(pool) < samples_per_concept).tolist())
        batches.append(np.asarray(idx))
    return batches


def clip_neighbor_groups(concept_cosine: np.ndarray, k: int) -> dict[int, list[int]]:
    """For each concept (row of the concept-concept CLIP cosine matrix), its `k` nearest OTHER concepts —
    the model-free hard negatives (semantically confusable in the shared CLIP target space). Self is excluded."""
    cosine = np.array(concept_cosine, dtype=float)
    np.fill_diagonal(cosine, -np.inf)                       # never pick self as its own neighbor
    return {concept: np.argsort(cosine[concept])[::-1][:k].tolist() for concept in range(cosine.shape[0])}
