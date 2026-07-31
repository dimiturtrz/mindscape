"""Unit test for the TF32 device switch (bd 62ak)."""
from neuroscan.perf import TorchPerf


def test_enable_fast_matmul_cpu_is_noop():
    TorchPerf.enable_fast_matmul("cpu")                     # no cuda side effects, must not raise
