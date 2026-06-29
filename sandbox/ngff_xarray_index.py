"""NGFFXarrayCoordinateSystem: xarray CoordinateTransform adapter for NGFF graphs."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr
from transformnd import TransformGraph


class NGFFXarrayCoordinateSystem(xr.indexes.CoordinateTransform):
    """Adapter that represents NGFF coordinate systems in xarray.

    Uses transformnd for path-finding and transform application.
    The ``forward`` / ``reverse`` methods satisfy the
    :class:`xarray.indexes.CoordinateTransform` protocol so that
    :class:`xarray.indexes.CoordinateTransformIndex` can use this class to
    lazily compute world-space coordinates on demand.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        shape: tuple[int, ...],
        tnd_graph: TransformGraph,
    ):
        self.target_coord_system = tnd_graph.coordinate_systems[target_name]
        try:
            self.transform_seq = tnd_graph.get_sequence(source_name, target_name)
            self.inverse_transform_seq = tnd_graph.get_sequence(target_name, source_name)
        except Exception as exc:
            raise ValueError(f"No transformation path found from {source_name!r} to {target_name!r}.") from exc
        self.graph = tnd_graph
        dim_names = [ax.name for ax in self.target_coord_system.axes]
        dim_shapes = {dim: shape[i] for i, dim in enumerate(dim_names)}
        super().__init__(
            coord_names=dim_names,
            dim_size=dim_shapes,
            dtype=np.dtype(float),
        )

    def _apply_seq(self, seq, coord_arrays: list) -> list:
        """Apply a TransformSequence to a list of per-dimension numpy arrays.

        transformnd expects coordinates as (N, D); we reshape, apply, then
        reshape back to the original spatial grid shape.
        """
        # transformnd.TransformGraph.get_sequence() finds the shortest path
        # and chains the edge transforms into a single TransformSequence.
        arrs = [np.asarray(a) for a in coord_arrays]
        shape = arrs[0].shape
        _coords = np.stack([a.ravel() for a in arrs], axis=1)
        result = seq.apply(_coords)
        return [result[:, i].reshape(shape) for i in range(result.shape[1])]

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        """Pixel → world coordinate transformation."""
        pixel = [dim_positions[dim] for dim in self.dims]
        world = self._apply_seq(self.transform_seq, pixel)
        return dict(zip(self.coord_names, world, strict=True))

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        """World → pixel coordinate transformation (used by xarray for sel())."""
        world = [coord_labels[name] for name in self.coord_names]
        pixel = self._apply_seq(self.inverse_transform_seq, world)
        return dict(zip(self.dims, pixel, strict=True))
