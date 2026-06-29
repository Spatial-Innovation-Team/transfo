"""ngff_transform: apply NGFF coordinate transforms to image data as xarray.

Code adapted from a notebook from @m-albert https://github.com/BioImageTools/ngff-transformations/blob/main/sandbox/xarray_representations_examples.ipynb

Two methods are provided:

  ``"resample"``
      Decompose the affine into a *rotation-like* part (rotation + shear +
      reflection) and a *scale + translation* part.  The rotation-like part is
      applied by resampling the pixel grid; scale and translation are stored as
      regular xarray axis coordinates.

  ``"lazy-indices"``
      Attach the *full* affine as a functional
      :class:`xarray.indexes.CoordinateTransformIndex`.  Pixel data stays
      untouched; world coordinates are computed on demand.

The ``compute`` parameter controls whether the underlying pixel array is
evaluated eagerly (``compute=True`` -> numpy) or kept as a lazy dask graph
(``compute=False``, the default).  For ``"resample"`` with ``compute=False``,
``dask_image.ndinterp.affine_transform`` is used; it is available via the
``dask`` optional-dependency group (``pip install transfo[dask]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import xarray as xr
from ngff_xarray_index import NGFFXarrayCoordinateSystem
from transformnd.base import TransformSequence

from transfo.affine_utils import decompose_affine

if TYPE_CHECKING:
    import dask.array as da
    from transformnd import TransformGraph

# Union of the two array types that can underlie an xr.DataArray.
# "da.Array" is a string forward reference so dask need not be imported at
# module load time.
_PixelArray = Union[np.ndarray, "da.Array"]


def transform(
    tnd_graph: TransformGraph,
    source: str,
    target: str,
    data: _PixelArray,
    *,
    method: Literal["resample", "lazy-indices"] = "lazy-indices",
    compute: bool = False,
) -> xr.DataArray:
    """Apply an NGFF coordinate transformation to image data.

    Parameters
    ----------
    tnd_graph
        Transformation graph built from OME-Zarr metadata (e.g. by
        :func:`ngff_graph.build_tnd_graph_from_image`).
    source
        Name of the source node in the graph; must be a pixel-indexed space
        (e.g. ``"s0"``), not a named coordinate system.  Passing a coordinate
        system as source is rejected to avoid ambiguity about which image to use.
    target
        Name of the target node.  Must be a named coordinate system in
        ``tnd_graph.coordinate_systems`` so that axis names are available.
    data
        Image data as a numpy or dask array.  Dimensions are assumed to be in
        the same order as the axes of *target*.
    method
        ``"lazy-indices"``  — keep pixel data unchanged, attach world
        coordinates as a functional index (no resampling).

        ``"resample"``  — resample the pixel grid to align with world axes,
        then store scale + translation as regular xarray coordinates.
    compute
        Whether to evaluate the pixel data eagerly.

        ``False`` (default) — the returned :class:`~xarray.DataArray` wraps a
        lazy dask graph.  ``dask_image.ndinterp.affine_transform`` is used for
        resampling (``pip install transfo[dask]``).

        ``True`` — the pixel array is evaluated to numpy before returning.
        For ``"resample"``: if the underlying array is already numpy,
        ``scipy.ndimage.affine_transform`` is used directly; if it is a dask
        array, ``dask_image.ndinterp.affine_transform`` is called and then
        ``.compute()`` is invoked (``pip install transfo[dask]``).

    Returns
    -------
    xarray.DataArray
        Image in world-coordinate space.  With ``"lazy-indices"`` the world
        coordinates are computed lazily; with ``"resample"`` they are stored
        as 1-D axis arrays.

    Raises
    ------
    ValueError
        If *source* or *target* are not nodes in the graph, if *source* is a
        named coordinate system, if *target* is not a named coordinate system,
        or the path cannot be represented as a single affine (for
        ``"resample"``).
    """
    if source not in tnd_graph.graph.nodes():
        raise ValueError(f"Source {source!r} is not a node in the transformation graph.")
    if source in tnd_graph.coordinate_systems:
        raise ValueError(
            f"Source {source!r} is a named coordinate system, not a pixel-indexed space. "
            "Pass a scale level (e.g. 's0') to avoid ambiguity about which image to use."
        )
    if target not in tnd_graph.graph.nodes():
        raise ValueError(f"Target {target!r} is not a node in the transformation graph.")
    if target not in tnd_graph.coordinate_systems:
        raise ValueError(
            f"Target {target!r} is not a named coordinate system. "
            "Axis names are required; pass a node that has named axes."
        )

    if method == "lazy-indices":
        return _lazy_indices(tnd_graph, source, target, data, compute=compute)
    if method == "resample":
        return _resample(tnd_graph, source, target, data, compute=compute)
    raise ValueError(f"Unknown method {method!r}. Expected 'resample' or 'lazy-indices'.")


def _resample(
    tnd_graph: TransformGraph,
    source: str,
    target: str,
    arr: _PixelArray,
    *,
    compute: bool,
) -> xr.DataArray:
    """Resample image into aligned world-axes; store scale+translation as xarray coords.

    Decomposition order:
        ``[rotation, shear, reflection, scale, translation]``

    The first three components form the *rotation-like* part ``P`` applied by
    resampling.  Scale and translation are stored as 1-D xarray coordinates so
    that ``DataArray.sel()`` works naturally.
    """
    ndim = arr.ndim

    # Obtain the single affine that captures the full source -> target path.
    seq = tnd_graph.get_sequence(source, target)
    full_affine = seq.to_affine()
    if full_affine is None:
        raise ValueError(
            f"The path from {source!r} to {target!r} cannot be expressed as a single affine. "
            "Use method='lazy-indices' instead. Future implementations will support resampling "
            "via non-linear transformations."
        )

    # Decompose: [rotation, shear, reflection, scale, translation]
    decomp = decompose_affine(full_affine, simple=False)
    rotation, shear, reflection, scale_t, translation_t = decomp

    # P = reflection \circ shear \circ rotation  (it's a "rotation-like" linear function with det != 0)
    # Note: with the "\circ" notation (mathematical notation), rotation is applied first
    # This function maps pixel -> intermediate space where scale+translation still apply.
    # Note: with the list-notation below, rotation is applied first
    linear_seq = TransformSequence([rotation, shear, reflection])
    linear_affine = linear_seq.to_affine()
    P = linear_affine.matrix[:ndim, :ndim]

    scale_values = np.asarray(scale_t.scale)
    translation_values = np.asarray(translation_t.translation)

    # Bounding box: transform all 2^ndim corners of the source pixel grid.
    # np.ndindex(2, 2, ...) gives all 0/1 combinations; multiplied by (shape-1)
    # this yields the actual corner positions in pixel space.
    corners = np.array(list(np.ndindex(*(2,) * ndim))) * (np.array(arr.shape) - 1)
    corners_t = corners @ P.T  # (2^ndim, ndim) in intermediate space
    lower = corners_t.min(axis=0)
    upper = corners_t.max(axis=0)
    # floor(upper) - floor(lower) + 1: the correct discrete span.
    # Using ceil(upper) is wrong when upper is exactly an integer — it gives
    # one too few pixels and silently drops the last input corner.
    output_shape = tuple((np.floor(upper) - np.floor(lower) + 1).astype(int).tolist())

    # prepare parameters to be used used by affine_transform()
    P_inv = np.linalg.inv(P)
    offset = P_inv @ lower

    is_dask = hasattr(arr, "compute")

    if compute and not is_dask:
        try:
            from scipy.ndimage import affine_transform as _scipy_affine_transform
        except ImportError as exc:
            raise ImportError(
                "scipy is required for eager resampling of numpy arrays. Install it with: pip install transfo[dask]"
            ) from exc
        image_t = _scipy_affine_transform(
            np.asarray(arr),
            matrix=P_inv,
            offset=offset,
            output_shape=output_shape,
            order=1,
            mode="constant",
            cval=0.0,
        )
    else:
        # dask input (compute=True or False) or lazy output (compute=False) → dask_image
        try:
            from dask_image.ndinterp import affine_transform as _dask_affine_transform
        except ImportError as exc:
            raise ImportError(
                "dask_image is required for resampling dask arrays or for lazy output. "
                "Install it with: pip install transfo[dask]"
            ) from exc
        if not is_dask:
            import dask.array as da

            arr = da.from_array(arr)
        image_t = _dask_affine_transform(
            arr,
            matrix=P_inv,
            offset=offset,
            output_shape=output_shape,
            order=1,
            mode="constant",
            cval=0.0,
        )
        if compute:
            image_t = image_t.compute()

    # Axis names from the named target coordinate system.
    axes = [ax.name for ax in tnd_graph.coordinate_systems[target].axes]

    # World coordinate for output pixel q[i] in dimension i:
    #   intermediate_i = q[i] + lower[i]
    #   world_i        = scale_i * intermediate_i + translation_i
    coords = {
        ax: (ax, (np.arange(output_shape[i]) + lower[i]) * scale_values[i] + translation_values[i])
        for i, ax in enumerate(axes)
    }
    return xr.DataArray(image_t, dims=axes, coords=coords)


def _lazy_indices(
    tnd_graph: TransformGraph,
    source: str,
    target: str,
    arr: _PixelArray,
    *,
    compute: bool,
) -> xr.DataArray:
    """Attach the full affine as a CoordinateTransformIndex.

    World-coordinate indices are always lazy (computed on demand by xarray).
    ``compute`` only controls whether the pixel data itself is a numpy array
    (``compute=True``) or a dask graph (``compute=False``).
    """
    shape = arr.shape
    pixel_data = arr.compute() if (compute and hasattr(arr, "compute")) else arr
    cs = NGFFXarrayCoordinateSystem(
        source_name=source,
        target_name=target,
        shape=shape,
        tnd_graph=tnd_graph,
    )
    index = xr.indexes.CoordinateTransformIndex(cs)
    coords = xr.Coordinates.from_xindex(index)
    return xr.DataArray(pixel_data, coords=coords)


##
