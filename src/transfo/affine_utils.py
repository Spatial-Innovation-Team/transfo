"""
Utilities for decomposing affine transformations into simpler components.

This code is adapted from the corresponding implementation in spaitaldata:
https://github.com/scverse/spatialdata/blob/471ae71dcc84754fe5c2cd48d9fe46a153ee2dee/src/spatialdata/transformations/transformations.py#L837
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from transformnd.base import Transform, TransformSequence
from transformnd.transforms import Affine, Scale, Translate
from transformnd.types import Spaces


def _compose_affine_from_linear_and_translation(
    linear: ArrayLike,
    translation: ArrayLike,
    *,
    spaces: Spaces = Spaces(None, None),
) -> Affine:
    """Build an Affine from a linear matrix and a translation vector."""
    linear = np.asarray(linear, dtype=float)
    translation = np.asarray(translation, dtype=float)
    n = linear.shape[0]
    matrix = np.zeros((n + 1, n + 1))
    matrix[:-1, :-1] = linear
    matrix[:-1, -1] = translation
    matrix[-1, -1] = 1.0
    return Affine(matrix, spaces=spaces)


def decompose_affine(
    transform: Transform,
    *,
    simple: bool = True,
) -> TransformSequence:
    """
    Decompose an affine transformation into a sequence of simpler transformations.

    Parameters
    ----------
    transform
        The transformation to decompose.  Must be representable as a square
        affine (i.e. ``transform.to_affine()`` must not return ``None``).
    simple
        If ``True``, split into a linear part followed by a translation.
        If ``False``, perform a full decomposition into reflection, rotation,
        shear, scale, and translation.

    Returns
    -------
    TransformSequence
        A sequence of transforms whose composition equals *affine*.
        Applied in list order (first element is applied to coordinates first).

        When ``simple=True`` the sequence contains:

        1. **Linear** (:class:`~transformnd.transforms.Affine`): the
           translation-free linear part of the affine.
        2. **Translation** (:class:`~transformnd.transforms.Translate`).

        When ``simple=False`` the sequence contains:

        1. **Rotation** (:class:`~transformnd.transforms.Affine`): orthogonal
           matrix with determinant 1.  For 2-D inputs this is a standard
           rotation; for higher dimensions it is an N-D orthogonal
           transformation.
        2. **Shear** (:class:`~transformnd.transforms.Affine`): upper-triangular
           matrix with 1s on the diagonal.
        3. **Reflection** (:class:`~transformnd.transforms.Scale`) with values
           in ``{-1, 1}``.  All values are 1 except possibly the first, which
           encodes whether the overall transformation includes a reflection.
        4. **Scale** (:class:`~transformnd.transforms.Scale`) with positive
           values.
        5. **Translation** (:class:`~transformnd.transforms.Translate`).

        Some components may be identity transformations.

    Raises
    ------
    ValueError
        If *transform* cannot be represented as an affine transformation, or
        if the resulting affine is not square (input and output dimensionalities differ).
    """
    affine = transform.to_affine()
    if affine is None:
        raise ValueError(f"{type(transform).__name__} cannot be represented as an affine transformation.")

    if affine.ndims.source != affine.ndims.target:
        raise ValueError(
            f"Only square affine transformations can be decomposed "
            f"(got {affine.ndims.source}D → {affine.ndims.target}D)."
        )

    matrix = affine.matrix
    ndim = affine.ndims.source

    translation_part = matrix[:-1, -1]
    linear_part = matrix[:-1, :-1]

    cond = np.linalg.cond(linear_part)
    if cond > 1e10:
        warnings.warn(
            f"The linear part of the affine has a large condition number ({cond:.2e}). "
            "The decomposition may be numerically inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    if simple:
        linear = _compose_affine_from_linear_and_translation(
            linear_part,
            np.zeros(ndim),
        )
        translation = Translate(translation_part)
        sequence = TransformSequence([linear, translation], spaces=affine.spaces)

    else:
        # RQ decomposition: linear_part = r @ q  (r upper-triangular, q orthogonal)
        r, q = scipy.linalg.rq(linear_part)

        # Ensure the diagonal of r is strictly positive.
        sign_diag = np.sign(np.diag(r))
        sign_diag[sign_diag == 0] = 1.0  # treat zero pivots as positive
        d = np.diag(sign_diag)
        r_pos = r @ d  # upper-triangular, positive diagonal
        q_adj = d @ q  # still orthogonal

        # Split r_pos into scale and shear.
        scale_values = np.diag(r_pos)  # all positive
        scale_matrix = np.diag(scale_values)
        shear_matrix = np.linalg.inv(scale_matrix) @ r_pos  # upper-tri, 1s on diag

        # Split q_adj into rotation (det = +1) and an axis-aligned reflection.
        # Reflection flips only the first axis when det(q_adj) = -1.
        det_sign = float(np.round(np.linalg.det(q_adj)))  # ±1
        reflection_values = np.ones(ndim)
        reflection_values[0] = det_sign
        reflection_matrix = np.diag(reflection_values)
        # q_adj = rotation_matrix @ reflection_matrix  →  rotation_matrix = q_adj @ reflection_matrix
        rotation_matrix = q_adj @ reflection_matrix  # det = det_sign * det_sign = 1

        # Conjugate rotation and shear by the reflection so the sequence becomes
        # [rotation', shear', reflection, scale, translation].  This lets callers
        # bundle the reflection with either the shear or the scale.
        # rotation' = reflection @ rotation @ reflection  (still orthogonal, det = 1)
        # shear'    = reflection @ shear    @ reflection  (still upper-tri, 1s on diag)
        rotation_matrix_adj = reflection_matrix @ rotation_matrix @ reflection_matrix
        shear_matrix_adj = reflection_matrix @ shear_matrix @ reflection_matrix

        if not np.allclose(
            scale_matrix @ reflection_matrix @ shear_matrix_adj @ rotation_matrix_adj,
            linear_part,
        ):
            raise RuntimeError("Affine decomposition failed internal consistency check. Please report this bug.")

        rotation = _compose_affine_from_linear_and_translation(rotation_matrix_adj, np.zeros(ndim))
        shear = _compose_affine_from_linear_and_translation(shear_matrix_adj, np.zeros(ndim))
        reflection = Scale(reflection_values)
        scale = Scale(scale_values)
        translation = Translate(translation_part)
        sequence = TransformSequence(
            [rotation, shear, reflection, scale, translation],
            spaces=affine.spaces,
        )

    reconstructed = sequence.to_affine()
    assert reconstructed is not None and np.allclose(reconstructed.matrix, matrix), (
        "Affine decomposition failed round-trip check."
    )

    return sequence
