from __future__ import annotations

import numpy as np
from transformnd import TransformGraph, TransformSequence
from transformnd.transforms import Affine, Scale, Translate


def _cs_id_to_str(identifier) -> str:
    """Encodes a CoordinateSystemIdentifier into string, escaping : into :: (internal).

    This is a temporary solution since it is introducing an arbitrary string
     representation and escape mechanism.
    """

    def escape_colon(s: str) -> str:
        return "" if s is None else s.replace(":", "::")

    if isinstance(identifier, str):
        return escape_colon(identifier)
    if isinstance(identifier.path, str) and isinstance(identifier.name, str):
        return f"{escape_colon(identifier.path)}:{escape_colon(identifier.name)}"
    if isinstance(identifier.path, str):
        return escape_colon(identifier.path)
    if isinstance(identifier.name, str):
        return escape_colon(identifier.name)
    raise ValueError(
        "The coordinate system identifier must either be a string or a"
        "CoordinateSystemIdentifier with a path and a name, where at least one is not "
        "None."
    )


def _ome_transform_to_tnd(transform):
    """Convert a single ome-zarr-models transform to a transformnd transform."""
    from ome_zarr_models.v06.coordinate_transforms import (
        Affine as OmeAffine,
    )
    from ome_zarr_models.v06.coordinate_transforms import (
        Scale as OmeScale,
    )
    from ome_zarr_models.v06.coordinate_transforms import (
        Sequence as OmeSequence,
    )
    from ome_zarr_models.v06.coordinate_transforms import (
        Translation as OmeTranslation,
    )

    if isinstance(transform, OmeScale):
        return Scale(np.array(transform.scale, dtype=float))

    if isinstance(transform, OmeTranslation):
        return Translate(np.array(transform.translation, dtype=float))

    if isinstance(transform, OmeAffine):
        # ome-zarr Affine is N×(N+1) augmented; transformnd expects (N+1)×(N+1) homogeneous.
        mat_aug = np.array(transform.affine_matrix, dtype=float)
        n = mat_aug.shape[0]
        mat_hom = np.eye(n + 1)
        mat_hom[:n, :] = mat_aug
        return Affine(mat_hom)

    if isinstance(transform, OmeSequence):
        return TransformSequence([_ome_transform_to_tnd(t) for t in transform.transformations])

    raise NotImplementedError(
        f"Cannot convert ome-zarr-models transform of type {type(transform).__name__!r} to a transformnd transform."
    )


def build_tnd_graph_from_image(image_attrs) -> TransformGraph:
    """Build a :class:`transformnd.TransformGraph` from an ome-zarr-models ImageAttrs.

    All transforms are added in both directions (forward + inverse) so that
    any source↔target path can be resolved.

    The returned graph carries an extra ``coordinate_systems`` attribute
    (``dict[str, CoordinateSystem]``) with the named coordinate systems from
    the ome-zarr multiscales metadata.

    Parameters
    ----------
    image_attrs
        :class:`ome_zarr_models.v06.image.ImageAttrs` instance (returned by
        :func:`datasets.zarr_to_image_attrs`).

    Returns
    -------
    TransformGraph
        transformnd graph with attached ``coordinate_systems`` dict.
    """
    tnd_graph = TransformGraph()
    cs_dict: dict = {}

    for multiscale in image_attrs.multiscales:
        for cs in multiscale.coordinateSystems:
            cs_dict[cs.name] = cs

        if multiscale.coordinateTransformations:
            for tx in multiscale.coordinateTransformations:
                tnd_tx = _ome_transform_to_tnd(tx)
                src, tgt = _cs_id_to_str(tx.input), _cs_id_to_str(tx.output)
                tnd_graph.add_transform(tnd_tx, source=src, target=tgt)
                tnd_graph.add_transform(tnd_tx.invert(), source=tgt, target=src)

        for dataset in multiscale.datasets:
            for tx in dataset.coordinateTransformations:
                tnd_tx = _ome_transform_to_tnd(tx)
                src, tgt = _cs_id_to_str(tx.input), _cs_id_to_str(tx.output)
                tnd_graph.add_transform(tnd_tx, source=src, target=tgt)
                tnd_graph.add_transform(tnd_tx.invert(), source=tgt, target=src)

    tnd_graph.coordinate_systems = cs_dict
    return tnd_graph


def tnd_graph_to_graphviz(graph: TransformGraph):
    """Return a :class:`graphviz.Digraph` visualising a transformnd TransformGraph.

    Nodes represent coordinate spaces. Edges are labeled with the transform
    type (e.g. ``Affine``, ``Scale``). Both forward and inverse edges are shown
    since they carry different semantic meanings.

    Parameters
    ----------
    graph
        The transformnd TransformGraph to visualise.

    Returns
    -------
    graphviz.Digraph
    """
    import graphviz
    from transformnd.graph import TRANSFORM_KEY

    dot = graphviz.Digraph(comment="transformnd Transform Graph")
    dot.attr(rankdir="LR")
    dot.attr("node", style="filled", fillcolor="#c6e5f5", shape="box", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    for node in graph.graph.nodes():
        dot.node(str(node), str(node))

    for src, tgt, data in graph.graph.edges(data=True):
        tx = data.get(TRANSFORM_KEY)
        label = type(tx).__name__ if tx is not None else ""
        dot.edge(str(src), str(tgt), label=label)

    return dot
