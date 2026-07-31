"""Batch-construction cores for contrastive retrieval — pure index/CLIP math."""
import numpy as np

from neuroscan.tasks.visual.sampling import BatchSpec, Sampling


def test_clip_hard_batches_seed_plus_neighbours_and_balanced():
    concept_ids = np.repeat(np.arange(4), 6)                # 4 concepts x 6
    neighbors = {0: [1, 2], 1: [0, 3], 2: [3, 0], 3: [2, 1]}
    batches = Sampling.clip_hard_batches(concept_ids, neighbors, BatchSpec(2, 3, n_batches=8), np.random.default_rng(0))
    for b in batches:
        counts = np.bincount(concept_ids[b])
        present = counts[counts > 0]
        assert len(present) == 2 and np.all(present == 3)   # 2 concepts, 3 each — balanced
        lo, hi = sorted(set(concept_ids[b].tolist()))
        assert hi in neighbors[lo] or lo in neighbors[hi]   # the pair is a seed + a CLIP-neighbour (hard)


def test_balanced_batches_equal_per_concept():
    concept_ids = np.repeat(np.arange(6), 10)               # 6 concepts x 10 trials
    batches = Sampling.balanced_batches(concept_ids, BatchSpec(3, 4, n_batches=5), np.random.default_rng(0))
    for b in batches:
        assert len(b) == 12                                 # 3 concepts x 4
        counts = np.bincount(concept_ids[b])
        present = counts[counts > 0]
        assert len(present) == 3 and np.all(present == 4)   # exactly 3 concepts, 4 trials each — STRICT balance


def test_balanced_batches_scarce_concept_uses_replacement():
    concept_ids = np.array([0, 0, 1])                       # concept 1 has only 1 trial
    batches = Sampling.balanced_batches(concept_ids, BatchSpec(2, 3, n_batches=1), np.random.default_rng(1))
    counts = np.bincount(concept_ids[batches[0]])
    assert np.all(counts[counts > 0] == 3)                  # still 3 each (concept 1 drawn with replacement)


def test_stratified_batches_span_distinct_concepts():
    # 4 concepts x 5 trials each; batch of 4 should hit ~4 distinct concepts (round-robin), not repeat one
    concept_ids = np.repeat(np.arange(4), 5)
    batches = Sampling.stratified_batches(concept_ids, batch_size=4, rng=np.random.default_rng(0))
    assert sum(len(b) for b in batches) == 20                    # every trial used exactly once
    assert np.array_equal(np.sort(np.concatenate(batches)), np.arange(20))
    # the first full batches span 4 distinct concepts (balanced), vs uniform which could draw all-same
    first = batches[0]
    assert len({int(concept_ids[i]) for i in first}) == 4


def test_stratified_uneven_concepts_still_covers_all():
    concept_ids = np.array([0, 0, 0, 1, 2])                      # imbalanced
    batches = Sampling.stratified_batches(concept_ids, batch_size=2, rng=np.random.default_rng(1))
    assert np.array_equal(np.sort(np.concatenate(batches)), np.arange(5))


def test_clip_neighbor_groups_nearest_excluding_self():
    # concept 0 closest to 2 then 1; concept 1 closest to 0; diagonal (self) must be excluded
    cos = np.array([[1.0, 0.3, 0.9],
                    [0.3, 1.0, 0.1],
                    [0.9, 0.1, 1.0]])
    groups = Sampling.clip_neighbor_groups(cos, k=1)
    assert groups[0] == [2]                                      # 0.9 > 0.3, self excluded
    assert groups[2] == [0]
    assert 1 not in groups[1][:0]                                # self never appears
    g2 = Sampling.clip_neighbor_groups(cos, k=2)
    assert g2[0] == [2, 1]                                       # ranked by cosine, self dropped
