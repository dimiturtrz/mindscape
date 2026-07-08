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

from dataclasses import dataclass

import numpy as np


@dataclass
class BatchSpec:
    """Batch geometry for the strict/hard samplers: each batch = `concepts_per_batch` × `samples_per_concept`
    trials; `n_batches` = how many per epoch (None -> one pass over the data)."""
    concepts_per_batch: int
    samples_per_concept: int
    n_batches: int | None = None


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


def _draw(pool: list[int], k: int, rng: np.random.Generator) -> list[int]:
    return rng.choice(pool, size=k, replace=len(pool) < k).tolist()


def balanced_batches(concept_ids: np.ndarray, spec: BatchSpec, rng: np.random.Generator) -> list[np.ndarray]:
    """STRICT balanced batches (bd 2j2): each batch = `spec.concepts_per_batch` concepts ×
    `spec.samples_per_concept` trials each, so every concept in the batch is represented EQUALLY (unlike
    `stratified_batches`' round-robin, which decays as scarce concepts empty out). Scarce concepts are drawn
    WITH replacement to keep the count exact. Balances the InfoNCE negatives across the label space every step."""
    by_concept: dict[int, list[int]] = {}
    for i, concept in enumerate(concept_ids):
        by_concept.setdefault(int(concept), []).append(i)
    concepts = list(by_concept)
    per_batch = min(spec.concepts_per_batch, len(concepts))
    n_batches = spec.n_batches or max(1, len(concept_ids) // (per_batch * spec.samples_per_concept))
    batches = []
    for _ in range(n_batches):
        chosen = rng.choice(concepts, size=per_batch, replace=False)
        idx = [i for concept in chosen for i in _draw(by_concept[int(concept)], spec.samples_per_concept, rng)]
        batches.append(np.asarray(idx))
    return batches


def clip_neighbor_groups(concept_cosine: np.ndarray, k: int) -> dict[int, list[int]]:
    """For each concept (row of the concept-concept CLIP cosine matrix), its `k` nearest OTHER concepts —
    the model-free hard negatives (semantically confusable in the shared CLIP target space). Self is excluded."""
    cosine = np.array(concept_cosine, dtype=float)
    np.fill_diagonal(cosine, -np.inf)                       # never pick self as its own neighbor
    return {concept: np.argsort(cosine[concept])[::-1][:k].tolist() for concept in range(cosine.shape[0])}


def clip_hard_batches(concept_ids: np.ndarray, neighbor_groups: dict[int, list[int]], spec: BatchSpec,
                      rng: np.random.Generator) -> list[np.ndarray]:
    """Balanced batches whose concepts are a SEED + its CLIP-nearest neighbours (bd 4ru) — semantically HARD
    negatives from epoch 0, model-free (the neighbours come from the CLIP target space, no encoder pass). Each
    batch: pick a seed, take it + its top neighbours present in the data (pad with random concepts if short),
    draw `spec.samples_per_concept` trials each (replacement for scarce), so it stays strictly balanced."""
    by_concept: dict[int, list[int]] = {}
    for i, concept in enumerate(concept_ids):
        by_concept.setdefault(int(concept), []).append(i)
    present = list(by_concept)
    per_batch = min(spec.concepts_per_batch, len(present))
    n_batches = spec.n_batches or max(1, len(concept_ids) // (per_batch * spec.samples_per_concept))
    batches = []
    for _ in range(n_batches):
        seed = int(rng.choice(present))
        group = list(dict.fromkeys([seed, *(c for c in neighbor_groups.get(seed, []) if c in by_concept)]))[:per_batch]
        while len(group) < per_batch:                      # pad if the seed has too few present neighbours
            extra = int(rng.choice(present))
            if extra not in group:
                group.append(extra)
        idx = [i for concept in group for i in _draw(by_concept[concept], spec.samples_per_concept, rng)]
        batches.append(np.asarray(idx))
    return batches
