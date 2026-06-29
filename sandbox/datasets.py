"""Dataset registry and loaders for the transfo sandbox.

Uses scverse-misc for downloading with pooch-backed caching and hash verification.
"""

from __future__ import annotations

from pathlib import Path

from scverse_misc.datasets import fetch, parse_registry, register_loader

_REGISTRY_PATH = Path(__file__).parent / "datasets.yaml"
_BASE_URL, _DATASETS = parse_registry(_REGISTRY_PATH)
_CACHE_DIR = Path.home() / ".cache" / "transfo"


@register_loader("zarr_metadata")
def _load_zarr_metadata(entry, target, download, **kwargs):
    """Download zarr metadata files (no chunk data) and open as a zarr.Group.

    File names in the registry encode relative paths within the zarr directory
    (e.g. ``s0/zarr.json``). pooch recreates the subdirectory tree under
    ``target / entry.name``.
    """
    import zarr

    zarr_dir = target / entry.name
    for file in entry.files:
        download(file, dest=zarr_dir)
    return zarr.open_group(str(zarr_dir), mode="r")


def fetch_affine_multiscale():
    """Download and cache affine_multiscale.zarr metadata and return a zarr.Group.

    Returns
    -------
    zarr.Group
        Root zarr group of the downloaded store.
    """
    return fetch(_DATASETS["affine_multiscale"], _CACHE_DIR, base_url=_BASE_URL)
