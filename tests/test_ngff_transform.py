"""Tests for sandbox/ngff_transform.py."""

import pytest


@pytest.mark.skip(
    reason=(
        "Requires NGFFScene support in ome-zarr-models-py. "
        "A Scene groups multiple images that share the same named coordinate system; "
        "the source validation in transform() (which rejects named coordinate systems "
        "as source to avoid ambiguity over which image to use) is the relevant guard. "
        "Test data: https://github.com/jo-mueller/ngff-rfc5-coordinate-transformation-examples"
        "/tree/cdb20ed60b35d441a85041eef079936cd5a895da/user_stories/human_organ_atlas.zarr"
    )
)
def test_transform_source_ambiguity_in_scene():
    """transform() must reject a array coordinate system passed as target.

    In an NGFFScene, multiple images are registered to the same named coordinate
    system.  Passing an coordinate system as ``target`` implies that a Scene is present.
    Once Scene is supported, this test should build a scene from ``human_organ_atlas.zarr``,
    attempt to call ``transform(scene_graph, source="<array_coord_sys>",
    target="<array_coord_sys_other_image>)``.
    """
    raise NotImplementedError
