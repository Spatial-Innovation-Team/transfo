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
    # `transform` demo

    Demonstrates both methods provided by `ngff_transform.py`:

    | Method | Pixel data | World coords |
    |---|---|---|
    | `"lazy-indices"` | unchanged | functional index (`CoordinateTransformIndex`) |
    | `"resample"` | resampled to world-aligned grid | stored as 1-D xarray coords |

    The `compute` parameter controls whether the pixel array is evaluated
    eagerly to numpy (`compute=True`) or kept as a lazy dask graph
    (`compute=False`, the default).  For `"resample"`, the backend depends on
    the input array type and `compute`:
    - numpy input + `compute=True` → `scipy.ndimage.affine_transform`
    - dask input or `compute=False` → `dask_image.ndinterp.affine_transform`

    The dataset is `affine_multiscale.zarr`, a small 3-D image with a shear +
    scale + translation transform from pixel space (`s0`) to world space
    (`sheared`).
    """)
    return


@app.cell
def _():
    import dask.array as da
    import numpy as np
    import xarray as xr
    from datasets import fetch_affine_multiscale
    from ngff_graph import build_tnd_graph_from_image, tnd_graph_to_graphviz
    from ngff_transform import transform
    from ome_zarr_models.v06.image import ImageAttrs

    return (
        ImageAttrs,
        build_tnd_graph_from_image,
        da,
        fetch_affine_multiscale,
        np,
        tnd_graph_to_graphviz,
        transform,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load dataset and build the transform graph
    """)
    return


@app.cell
def _(ImageAttrs, build_tnd_graph_from_image, fetch_affine_multiscale):
    zarr_group = fetch_affine_multiscale()
    image_attrs = ImageAttrs.model_validate(dict(zarr_group.attrs)["ome"])
    tnd_graph = build_tnd_graph_from_image(image_attrs)
    return tnd_graph, zarr_group


@app.cell
def _(tnd_graph, tnd_graph_to_graphviz):
    # Transform graph: s0/s1/s2 → sheared (world coordinate system)
    tnd_graph_to_graphviz(tnd_graph)
    return


@app.cell
def _(da, np, zarr_group):
    shape_s0 = zarr_group["s0"].shape  # (27, 226, 186) — z, y, x
    shape_s1 = zarr_group["s1"].shape  # (13, 113, 93)  — z, y, x
    rng = np.random.default_rng(42)
    array_data_s0 = da.from_array(
        rng.integers(0, 255, shape_s0, dtype=np.uint8),
        chunks=shape_s0,
    )
    array_data_s1 = da.from_array(
        rng.integers(0, 255, shape_s1, dtype=np.uint8),
        chunks=shape_s1,
    )
    print(f"s0 array shape: {shape_s0}  (z, y, x)")
    print(f"s1 array shape: {shape_s1}  (z, y, x)")
    return array_data_s0, array_data_s1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method `"lazy-indices"` — functional world-coordinate index

    The pixel data is unchanged; world-space coordinates are attached as a
    `CoordinateTransformIndex` and computed on demand.

    ### `compute=False` (default) — pixel data stays as a dask array
    """)
    return


@app.cell
def _(array_data_s0, tnd_graph, transform):
    xim_li_lazy = transform(
        tnd_graph,
        source="s0",
        target="sheared",
        data=array_data_s0,
        method="lazy-indices",
        compute=False,
    )
    xim_li_lazy  # noqa: B018
    return (xim_li_lazy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `compute=True` — pixel data evaluated to numpy
    """)
    return


@app.cell
def _(array_data_s0, tnd_graph, transform):
    xim_li_eager = transform(
        tnd_graph,
        source="s0",
        target="sheared",
        data=array_data_s0,
        method="lazy-indices",
        compute=True,
    )
    xim_li_eager  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point-wise world-coordinate lookup via `sel()`

    `xarray` calls the index's `reverse()` to map world coords back to pixel
    indices, then retrieves the nearest value.
    """)
    return


@app.cell
def _(xim_li_lazy, xr):
    # Values must be wrapped in xr.DataArray so xarray performs vectorised
    # (paired) point lookup rather than an outer product.  The dimension name
    # is arbitrary; omitting dims= uses the default "dim_0".
    # pixel (5, 50, 30) → sheared ≈ (50.0, 150.0, 60.0)
    _query = {
        "z": xr.DataArray([50.0, 100.0]),
        "y": xr.DataArray([150.0, 300.0]),
        "x": xr.DataArray([60.0, 120.0]),
    }
    xim_li_lazy.sel(**_query, method="nearest")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method `"resample"` — pixel grid resampled into world space

    The affine is decomposed into:
    - **rotation + shear + reflection** → applied by resampling the pixel grid
    - **scale + translation** → stored as 1-D xarray coordinates

    ### `compute=False` (default) — lazy resampling via `dask_image`
    """)
    return


@app.cell
def _(array_data_s0, tnd_graph, transform):
    xim_rs_lazy = transform(
        tnd_graph,
        source="s0",
        target="sheared",
        data=array_data_s0,
        method="resample",
        compute=False,  # uses dask_image.ndinterp.affine_transform
    )
    xim_rs_lazy  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `compute=True` — eager resampling (scipy for numpy input, dask_image for dask input)
    """)
    return


@app.cell
def _(array_data_s0, tnd_graph, transform):
    xim_rs_s0_eager = transform(
        tnd_graph,
        source="s0",
        target="sheared",
        data=array_data_s0,
        method="resample",
        compute=True,  # dask input → dask_image + .compute(); numpy input → scipy
    )
    xim_rs_s0_eager  # noqa: B018
    return (xim_rs_s0_eager,)


@app.cell
def _(array_data_s1, tnd_graph, transform):
    xim_rs_s1_eager = transform(
        tnd_graph,
        source="s1",
        target="sheared",
        data=array_data_s1,
        method="resample",
        compute=True,  # dask input → dask_image + .compute(); numpy input → scipy
    )
    xim_rs_s1_eager  # noqa: B018
    return (xim_rs_s1_eager,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisation

    Plot a single z-slice.  Axis ticks show world-space coordinates from the
    stored 1-D coordinate arrays.
    """)
    return


@app.cell
def _(xim_rs_s0_eager, xim_rs_s1_eager):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    z_mid_s0 = xim_rs_s0_eager.shape[0] // 2
    xim_rs_s0_eager.isel(z=z_mid_s0).plot(ax=axes[0])
    axes[0].set_title(f"Resampled — z slice {z_mid_s0} scale 0 (world coords)")

    z_mid_s1 = xim_rs_s1_eager.shape[0] // 2
    xim_rs_s1_eager.isel(z=z_mid_s1).plot(ax=axes[1])
    axes[1].set_title(f"Resampled — z slice {z_mid_s1} scale 1 (world coords)")
    axes[1].set_xlim(xim_rs_s0_eager.x.min(), xim_rs_s0_eager.x.max())
    axes[1].set_ylim(xim_rs_s0_eager.y.min(), xim_rs_s0_eager.y.max())

    plt.tight_layout()
    plt.show()

    for dim in ["z", "y", "x"]:
        coords_s0 = xim_rs_s0_eager.coords[dim].values
        print(f"  {dim} scale0: [{coords_s0[0]:.2f}, {coords_s0[-1]:.2f}]  ({len(coords_s0)} pixels)")

        coords_s1 = xim_rs_s1_eager.coords[dim].values
        print(f"  {dim} scale1: [{coords_s1[0]:.2f}, {coords_s1[-1]:.2f}]  ({len(coords_s1)} pixels)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | | `"lazy-indices"` | `"resample"` |
    |---|---|---|
    | Pixel data resampled | No | Yes |
    | World coords | Functional index (always lazy) | 1-D arrays |
    | `compute=False` data backend | dask (input as-is) | dask (`dask_image`) |
    | `compute=True` data backend | numpy | numpy (`scipy` or `dask_image` + `.compute()`) |
    | `sel()` support | Point-wise | Regular xarray |
    | Extra deps (`transfo[dask]`) | None | `dask_image` + `scipy` |
    """)
    return


if __name__ == "__main__":
    app.run()
