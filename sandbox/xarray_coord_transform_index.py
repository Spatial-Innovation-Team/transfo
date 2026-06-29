import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NGFF transforms in xarray wth CoordinateTransformIndex

    Based on a notebook by [Marvin Albert](https://github.com/m-albert): https://github.com/BioImageTools/ngff-transformations/blob/main/sandbox/xarray_coord_transform_index.ipynb

    Overview: https://xarray-indexes.readthedocs.io/blocks/transform.html
    Further links:
    - https://docs.xarray.dev/en/latest/generated/xarray.indexes.CoordinateTransformIndex.html#xarray.indexes.CoordinateTransformIndex
    - https://docs.xarray.dev/en/latest/generated/xarray.indexes.CoordinateTransform.html#xarray.indexes.CoordinateTransform

    More xarray functionality nicely shown here: https://xarray-indexes.readthedocs.io/index.html
    """)
    return


@app.cell
def _():
    from collections.abc import Hashable
    from typing import Any

    import dask.array as da
    import numpy as np
    import xarray as xr
    from datasets import fetch_affine_multiscale
    from ngff_graph import build_tnd_graph_from_image, tnd_graph_to_graphviz
    from ome_zarr_models.v06.image import ImageAttrs
    from transformnd import TransformGraph

    return (
        Any,
        Hashable,
        ImageAttrs,
        TransformGraph,
        build_tnd_graph_from_image,
        da,
        fetch_affine_multiscale,
        np,
        tnd_graph_to_graphviz,
        xr,
    )


@app.cell
def _(Any, Hashable, TransformGraph, np, xr):
    class NGFFXarrayCoordinateSystem(xr.indexes.CoordinateTransform):
        """
        Adapter class that represents NGFF coordinate systems in xarray.

        Uses ome-zarr-models-py for the coordinate system model and
        transformnd for path-finding and transform application.
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
                raise ValueError(f"No transformation path found from {source_name} to {target_name}.") from exc
            self.graph = tnd_graph
            dim_names = [ax.name for ax in self.target_coord_system.axes]
            dim_shapes = {dim: shape[i] for i, dim in enumerate(dim_names)}
            super().__init__(
                coord_names=dim_names,
                dim_size=dim_shapes,
                dtype=np.dtype(float),
            )

        # transformnd.TransformGraph.get_sequence() finds the shortest path
        def _apply_seq(
            self, seq, coord_arrays: list
        ) -> list:  # and chains the edge transforms into a single TransformSequence.
            """
            Apply a TransformSequence to a list of per-dimension numpy arrays.

            transformnd expects coordinates as (N, D); we reshape, apply, then
            reshape back to the original spatial grid shape.
            """
            arrs = [np.asarray(a) for a in coord_arrays]
            shape = arrs[0].shape
            _coords = np.stack([a.ravel() for a in arrs], axis=1)
            result = seq.apply(_coords)
            return [result[:, i].reshape(shape) for i in range(result.shape[1])]

        def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
            """Perform array -> coordinate system transformation."""
            pixel = [dim_positions[dim] for dim in self.dims]
            world = self._apply_seq(self.transform_seq, pixel)
            return dict(zip(self.coord_names, world, strict=True))

        def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
            """Perform coordinate system -> array coordinate reverse transformation."""
            world = [coord_labels[name] for name in self.coord_names]
            pixel = self._apply_seq(self.inverse_transform_seq, world)
            return dict(zip(self.dims, pixel, strict=True))

    return (NGFFXarrayCoordinateSystem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading real data from affine_multiscale.zarr
    """)
    return


@app.cell
def _(ImageAttrs, fetch_affine_multiscale):
    zarr_group = fetch_affine_multiscale()
    image_attrs = ImageAttrs.model_validate(dict(zarr_group.attrs)["ome"])

    print(image_attrs)
    # json.loads(image_attrs.model_dump_json())
    return image_attrs, zarr_group


@app.cell
def _(image_attrs):
    omz_graph = image_attrs.transform_graph()
    omz_graph.to_graphviz()
    return


@app.cell
def _(build_tnd_graph_from_image, da, image_attrs, np, zarr_group):
    tnd_graph = build_tnd_graph_from_image(image_attrs)

    source_name = "s0"
    target_name = "sheared"

    array_shape = zarr_group["s0"].shape  # (27, 226, 186) — z, y, x (metadata only, no chunks)
    array_data = da.from_array(
        np.random.default_rng(0).integers(0, 255, array_shape, dtype=np.uint8),
        chunks=array_shape,
    )
    return array_data, array_shape, source_name, target_name, tnd_graph


@app.cell
def _(tnd_graph, tnd_graph_to_graphviz):
    tnd_graph_to_graphviz(tnd_graph)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Instanciating a xarray.DataArray with CoordinateTransformIndex
    """)
    return


@app.cell
def _(
    NGFFXarrayCoordinateSystem,
    array_data,
    array_shape,
    source_name,
    target_name,
    tnd_graph,
    xr,
):
    ngff_xarray_coord_system_transform = NGFFXarrayCoordinateSystem(
        source_name=source_name,
        target_name=target_name,
        shape=array_shape,
        tnd_graph=tnd_graph,
    )
    _index = xr.indexes.CoordinateTransformIndex(ngff_xarray_coord_system_transform)
    _index.to_pandas_index = lambda: (_ for _ in ()).throw(
        TypeError("Cannot convert NGFF coordinate system transform to pandas Index.")
    )
    _coords = xr.Coordinates.from_xindex(_index)
    xim = xr.DataArray(array_data, coords=_coords)
    return ngff_xarray_coord_system_transform, xim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## coordinates plot
    """)
    return


@app.cell
def _(ngff_xarray_coord_system_transform, np, xim):
    import matplotlib.projections as _mp
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    _mp.register_projection(Axes3D)

    # extract coordinates
    pts = np.array([ngff_xarray_coord_system_transform.generate_coords()[dim] for dim in xim.dims]).reshape(
        len(xim.dims), -1
    )
    pts = pts[:, ::50]

    # 3d matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[0], pts[1], pts[2])
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_zlabel("x")
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dask array
    """)
    return


@app.cell
def _(array_data):
    array_data  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dask + xarray
    """)
    return


@app.cell
def _(array_data, array_shape, np, xr):
    xim2 = xr.DataArray(
        array_data,
        dims=["z", "y", "x"],
        coords={dim: np.arange(size) * 0.1 for dim, size in zip(["z", "y", "x"], array_shape, strict=True)},
    )
    xim2  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dask + xarray + NGFFXarrayCoordinateSystem
    """)
    return


@app.cell
def _(
    NGFFXarrayCoordinateSystem,
    array_data,
    array_shape,
    source_name,
    target_name,
    tnd_graph,
    xr,
):
    ngff_xarray_coord_system_transform_1 = NGFFXarrayCoordinateSystem(
        source_name=source_name,
        target_name=target_name,
        shape=array_shape,
        tnd_graph=tnd_graph,
    )
    _index = xr.indexes.CoordinateTransformIndex(ngff_xarray_coord_system_transform_1)
    _coords = xr.Coordinates.from_xindex(_index)
    xim_ngff = xr.DataArray(array_data, coords=_coords)
    return (xim_ngff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transformation graph can be extracted from xarray DataArray

    ... without using attributes
    """)
    return


@app.cell
def _(xim_ngff):
    xim_ngff._indexes["x"].transform.target_coord_system  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downstream processing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### "classical" xarray.DataArray
    """)
    return


@app.cell
def _(plt, xim):
    _xim_plot = xim.isel(y=0)
    plt.figure()
    _xim_plot.plot.pcolormesh(x="x", y="z")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### "functional" coordinate arrays + xarray
    """)
    return


@app.cell
def _(plt, xim_ngff):
    _xim_plot = xim_ngff.isel(y=0)
    plt.figure()
    _xim_plot.plot.pcolormesh(x="x", y="z")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## xarray sel with CoordinateTransformIndex

    As of xarray 2025.10.1 this is supported with `method='nearest'`, but only for
    **point-wise** (vectorised) indexing — labels must be `xr.DataArray` or `xr.Variable`
    objects, not plain scalars or slices.
    Each entry is a world-space (sheared) coordinate; xarray calls `reverse()` internally
    to map it back to the nearest pixel index.
    """)
    return


@app.cell
def _(xim_ngff):
    xim_ngff  # noqa: B018
    return


@app.cell
def _(xim_ngff, xr):
    _query = {
        "z": xr.DataArray([50.0, 100.0], dims="points"),
        "y": xr.DataArray([150.0, 300.0], dims="points"),
        "x": xr.DataArray([60.0, 120.0], dims="points"),
    }
    # pixel (5, 50, 30)  → sheared (50, 150, 60)
    # pixel (10, 100, 60) → sheared (100, 300, 120)
    xim_ngff.sel(**_query, method="nearest")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Not supported

    Related discussion: https://github.com/pydata/xarray/issues/10572
    """)
    return


@app.cell
def _(xim_ngff):
    _query = {"z": slice(0, 10), "y": slice(0, 10), "x": slice(0, 10)}
    xim_ngff.sel(**_query, method="nearest")
    return


if __name__ == "__main__":
    app.run()
