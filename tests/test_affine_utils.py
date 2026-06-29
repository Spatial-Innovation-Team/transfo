"""Tests for affine decomposition utilities."""

import numpy as np
import pytest
from transformnd.base import Transform
from transformnd.transforms import Affine, Scale, Translate
from transformnd.types import NDims

from transfo.affine_utils import decompose_affine


class _NonAffineTransform(Transform):
    """Minimal concrete Transform whose to_affine() returns None.

    The base Transform.to_affine() already returns None, so no override is needed.
    """

    def __init__(self, ndim: int = 2):
        super().__init__(NDims(ndim, ndim))

    def apply(self, coords):
        return coords

    def invert(self):
        return self


def _make_affine(linear: np.ndarray, translation: np.ndarray | None = None) -> Affine:
    # Builds the (n+1)×(n+1) augmented homogeneous matrix expected by Affine.
    n = linear.shape[0]
    m = np.eye(n + 1)
    m[:-1, :-1] = linear
    if translation is not None:
        m[:-1, -1] = translation
    return Affine(m.astype(float))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# a round-trip test is performed by decompose_affine() at each run (we do manual matrix
# multiplication), here we rely on to the transformnd .to_affine() API
def _round_trip(affine: Affine, *, simple: bool) -> None:
    """Assert that decompose_affine reconstructs the original matrix."""
    seq = decompose_affine(affine, simple=simple)
    reconstructed = seq.to_affine()
    assert reconstructed is not None
    assert np.allclose(reconstructed.matrix, affine.matrix)


# ---------------------------------------------------------------------------
# simple=True (linear + translation)
# ---------------------------------------------------------------------------


class TestSimpleDecomposition:
    def test_identity_2d(self):
        a = _make_affine(linear=np.eye(2))
        seq = decompose_affine(a, simple=True)
        assert len(seq) == 2
        _round_trip(a, simple=True)

    def test_pure_translation_2d(self):
        a = _make_affine(linear=np.eye(2), translation=np.array([3.0, -7.0]))
        seq = decompose_affine(a, simple=True)
        linear, translation = seq[0], seq[1]
        assert isinstance(linear, Affine)
        assert isinstance(translation, Translate)
        assert np.allclose(linear.matrix[:-1, :-1], np.eye(2))
        assert np.allclose(translation.translation, [3.0, -7.0])

    def test_general_affine_2d(self):
        _round_trip(
            _make_affine(linear=np.array([[2.0, 0.5], [0.0, 3.0]]), translation=np.array([1.0, 2.0])), simple=True
        )

    def test_general_affine_3d(self):
        rng = np.random.default_rng(0)
        linear = rng.standard_normal((3, 3))
        # A @ A.T is symmetric positive semi-definite; adding I shifts all eigenvalues to ≥ 1,
        # guaranteeing invertibility.
        linear = linear @ linear.T + np.eye(3)
        _round_trip(_make_affine(linear=linear, translation=np.array([1.0, -2.0, 3.0])), simple=True)

    def test_non_square_raises(self):
        m = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="square"):
            decompose_affine(Affine(m), simple=True)

    def test_non_affine_transform_raises(self):
        with pytest.raises(ValueError, match="cannot be represented as an affine"):
            decompose_affine(_NonAffineTransform())

    def test_ill_conditioned_warns(self):
        # condition number ≈ 1e12, well above the 1e10 warning threshold
        linear = np.diag([1.0, 1e-12])
        with pytest.warns(RuntimeWarning, match="condition number"):
            decompose_affine(_make_affine(linear=linear))


# ---------------------------------------------------------------------------
# simple=False (rotation, shear, reflection, scale, translation)
# ---------------------------------------------------------------------------


class TestFullDecomposition:
    def test_returns_five_transforms(self):
        seq = decompose_affine(_make_affine(linear=np.eye(2), translation=np.array([1.0, 2.0])), simple=False)
        assert len(seq) == 5

    def test_component_types(self):
        rng = np.random.default_rng(1)
        q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        linear = np.diag([2.0, 3.0]) @ np.array([[1.0, 0.4], [0.0, 1.0]]) @ q
        seq = decompose_affine(_make_affine(linear=linear, translation=np.array([5.0, -1.0])), simple=False)
        rotation, shear, reflection, scale, translation = seq
        assert isinstance(rotation, Affine)
        assert isinstance(shear, Affine)
        assert isinstance(reflection, Scale)
        assert isinstance(scale, Scale)
        assert isinstance(translation, Translate)

    def test_reflection_values_in_pm1(self):
        seq = decompose_affine(_make_affine(linear=np.diag([-1.0, 2.0])), simple=False)
        reflection = seq[2]
        assert np.all(np.abs(reflection.scale) == 1.0)

    def test_scale_values_positive(self):
        rng = np.random.default_rng(2)
        linear = rng.standard_normal((2, 2))
        # Reject near-singular draws so the decomposition is numerically stable.
        while abs(np.linalg.det(linear)) < 0.1:
            linear = rng.standard_normal((2, 2))
        seq = decompose_affine(_make_affine(linear=linear), simple=False)
        scale = seq[3]
        assert np.all(scale.scale > 0)

    def test_rotation_determinant_one(self):
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        seq = decompose_affine(_make_affine(linear=q), simple=False)
        rotation = seq[0]
        rot_linear = rotation.matrix[:-1, :-1]
        assert np.isclose(np.linalg.det(rot_linear), 1.0)

    def test_round_trip_pure_scale(self):
        _round_trip(_make_affine(linear=np.diag([2.0, 3.0])), simple=False)

    def test_round_trip_general_2d(self):
        rng = np.random.default_rng(4)
        linear = rng.standard_normal((2, 2))
        # Reject near-singular draws so the decomposition is numerically stable.
        while abs(np.linalg.det(linear)) < 0.1:
            linear = rng.standard_normal((2, 2))
        _round_trip(_make_affine(linear=linear, translation=np.array([3.0, -1.0])), simple=False)

    def test_round_trip_with_reflection_2d(self):
        # Transformation that flips the x-axis
        _round_trip(_make_affine(linear=np.diag([-1.0, 1.0]), translation=np.array([1.0, 0.0])), simple=False)

    def test_round_trip_3d(self):
        rng = np.random.default_rng(5)
        linear = rng.standard_normal((3, 3))
        # Reject near-singular draws so the decomposition is numerically stable.
        while abs(np.linalg.det(linear)) < 0.1:
            linear = rng.standard_normal((3, 3))
        _round_trip(_make_affine(linear=linear, translation=np.array([1.0, 2.0, 3.0])), simple=False)
