"""Reproducible runs: seed every RNG the rig path draws from, then restore them.

The rig path samples from Python's ``random``, NumPy's global state, torch (CPU and
CUDA: ``do_sample`` in ``generate``, farthest-point start) and ``np.random.default_rng()``
built without a seed (VAE feature sampling). ``seeded(seed)`` fixes all of them for the
duration of the block and puts the caller's state back afterwards, so other ComfyUI
nodes are unaffected. A negative seed leaves everything as it is (upstream behaviour).

GPU kernels may still be nondeterministic across driver or hardware changes; on one
machine the same seed reproduces the same skeleton.
"""

from __future__ import annotations

import contextlib
import random

import numpy as np
import torch


@contextlib.contextmanager
def seeded(seed: int):
    if seed is None or int(seed) < 0:
        yield
        return
    seed = int(seed)
    py_state = random.getstate()
    np_state = np.random.get_state()
    orig_default_rng = np.random.default_rng
    # Each unseeded default_rng() gets its own child of one seeded sequence, so two
    # calls still differ from each other while the whole run is reproducible.
    children = np.random.SeedSequence(seed)

    def default_rng(seed_arg=None):
        if seed_arg is None:
            return orig_default_rng(children.spawn(1)[0])
        return orig_default_rng(seed_arg)

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        np.random.default_rng = default_rng
        try:
            yield
        finally:
            np.random.default_rng = orig_default_rng
            random.setstate(py_state)
            np.random.set_state(np_state)
