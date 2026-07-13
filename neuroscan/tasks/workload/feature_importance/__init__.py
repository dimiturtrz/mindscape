"""fNIRS feature-importance studies — *which* descriptor families carry the n-back workload signal.

Three angles on one question, sharing the wide descriptor bank (`core.features.DescriptorBank`) and the
subject-grouped CV plumbing (`._cv`):

  - `optuna_search`   — Optuna as a wrapper feature-selector: per-family weights, fANOVA importance + top-trial
                        weights + cross-seed stability. The importance *ranking* (robust) vs peak acc (optimistic).
  - `differentiable`  — the same idea by gradient descent (torch/CUDA): softmax weights + linear head, entropy(w)
                        as a differentiable sparsity penalty; sweep lambda → accuracy vs effective-#-features.
  - `recipes`         — the leakage-free confirm: fixed family sets under plain grouped CV (no search, no optimism),
                        the apples-to-apples table that anchors the README numbers.
"""
