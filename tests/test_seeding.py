import random

import numpy as np
import torch

from skintokens.seeding import seeded


def _draw():
    return (
        random.random(),
        float(np.random.rand()),
        float(np.random.default_rng().random()),
        float(np.random.default_rng().random()),
        float(torch.rand(1)),
    )


def test_same_seed_same_draws():
    with seeded(7):
        a = _draw()
    with seeded(7):
        b = _draw()
    assert a == b
    assert a[2] != a[3], "two unseeded default_rng() calls still differ inside one run"


def test_other_seed_other_draws():
    with seeded(7):
        a = _draw()
    with seeded(8):
        b = _draw()
    assert a != b


def test_restores_caller_state():
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    expected = _draw()[:2] + (float(torch.rand(1)),)
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    with seeded(99):
        _draw()
    got = _draw()[:2] + (float(torch.rand(1)),)
    assert got == expected
    assert np.random.default_rng.__module__.startswith("numpy")


def test_explicit_seeds_pass_through():
    with seeded(7):
        a = np.random.default_rng(seed=0).random()
        b = np.random.default_rng(0).random()
    assert a == b == np.random.default_rng(0).random()


def test_negative_seed_is_a_no_op():
    orig = np.random.default_rng
    with seeded(-1):
        assert np.random.default_rng is orig
