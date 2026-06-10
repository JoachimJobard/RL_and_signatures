"""Tests for the path-signature feature map (SlidingSignatureJAX).

Covers the truncated-signature size, output shape, the depth-1 = total-increment
identity, translation invariance of the un-augmented signature (the property that
makes the Perez Arribas origin-ramp sign immaterial, code_review F-C1), and that
the float32 and float64 paths agree to within float32 precision (F-E1).
"""

import numpy as np

from src.utils.dynamic_signature import SlidingSignatureJAX


def _signature_of(path, depth, d, *, time_aug=False, origin_aug=False, dtype=np.float64):
    sig = SlidingSignatureJAX(
        depth=depth, window_size=len(path) - 1, d=d,
        time_augmentation=time_aug, origin_augmentation=origin_aug,
        bias=False, use_jax_buffer=False, dtype=dtype,
    )
    sig.reset(prefill_zeros=False)
    for row in path:
        sig.append(np.asarray(row, dtype=float))
    return np.asarray(sig.current_signature, dtype=np.float64)


def test_signature_size_plain():
    sig = SlidingSignatureJAX(depth=3, window_size=10, d=2,
                              time_augmentation=False, origin_augmentation=False)
    assert sig.signature_size == 2 + 2 ** 2 + 2 ** 3  # sum d^i, i=1..3


def test_signature_size_with_augmentations():
    # effective channels d_eff = d + time(1) + origin(d) = 2*d + 1
    d = 2
    sig = SlidingSignatureJAX(depth=2, window_size=10, d=d,
                              time_augmentation=True, origin_augmentation=True)
    d_eff = 2 * d + 1
    assert sig.signature_size == d_eff + d_eff ** 2


def test_output_shape_matches_signature_size():
    path = np.linspace(0, 1, 6).reshape(-1, 1)
    sig = SlidingSignatureJAX(depth=3, window_size=5, d=1,
                              time_augmentation=True, origin_augmentation=True)
    sig.reset(prefill_zeros=False)
    for row in path:
        sig.append(row)
    out = np.asarray(sig.current_signature)
    assert out.shape == (sig.signature_size,)


def test_depth_one_signature_equals_total_increment():
    # Level-1 signature of a path is its total increment X_last - X_first.
    path = np.array([[0.2], [0.5], [-0.3], [1.1]])
    s = _signature_of(path, depth=1, d=1)
    assert np.allclose(s, path[-1] - path[0], atol=1e-10)


def test_unaugmented_signature_is_translation_invariant():
    # Without augmentation the signature depends only on increments, so adding a
    # constant to the whole path leaves it unchanged.
    rng = np.random.default_rng(0)
    path = rng.standard_normal((8, 2))
    shifted = path + np.array([3.0, -2.0])
    s1 = _signature_of(path, depth=3, d=2)
    s2 = _signature_of(shifted, depth=3, d=2)
    assert np.allclose(s1, s2, atol=1e-8)


def test_float32_and_float64_agree_to_float32_precision():
    rng = np.random.default_rng(1)
    path = rng.standard_normal((20, 1))
    s64 = _signature_of(path, depth=3, d=1, time_aug=True, origin_aug=True, dtype=np.float64)
    s32 = _signature_of(path, depth=3, d=1, time_aug=True, origin_aug=True, dtype=np.float32)
    rel = np.linalg.norm(s32 - s64) / np.linalg.norm(s64)
    assert rel < 1e-4
