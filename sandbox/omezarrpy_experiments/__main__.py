
import numpy as np

import ome_zarr as oz
from ome_zarr.scale import Methods
from ome_zarr import OMEZarrImage, OMEZarrMultiscale
from ome_zarr.classes.image import Axis, CoordinateSystem, CoordinateSystemIdentifier, MultiscaleV06, Translation
from ome_zarr.scale import Methods
import pydantic

import transformnd as tnd


# BASIC USAGE ###################

# create some random data to write
size_xy = 128
size_z = 10
rng = np.random.default_rng(0)
data = rng.poisson(lam=10, size=(2, size_z, size_xy, size_xy)).astype(np.uint8)


image = OMEZarrImage(
  data=data,
  axes=["c", "z", "y", "x"],
  scale={"c": 1.0, "z": 0.5, "y": 0.1, "x": 0.1},
)

# OMEZarrImage will enforce that image.scale is a dict[str, float], but this
# isn't communicated to the type hints, so the following produces a typing error:
img_scale: dict[str, float] = image.scale

# creating the multiscales creates a pyramid of (lazy) dask arrays that downsample the image
multiscales = OMEZarrMultiscale(
    image=image,
    scale_factors=(2, 4, 8),
    method="resize",
)

# also noteworthy is that ome-zarr-py, as well as ngff-zarr,
# assume that ome.multiscales is of length 1


# DOWNSCALING ##################

# It seems to me that one can only manipulate at the granularity of an
# entire image.
# I also think that the downscaling is _always_ performed bhy the ome_zarr library,
# rather than being left open to the user:


# You either pass the top level image to the OMEZarrMultiscale class, and
# it will auto-create (lazy) downscales of it, whicih then get materialized on save:
# OMEZarrMultiscale(image=image, method=Methods.NEAREST).to_ome_zarr("/tmp/auto_multiscale1.zarr")


# or you call write_image directly on some array-like data, which also just expects the top level image
# and will do the downscaling automatically:
from ome_zarr.writer import write_image
write_image(
    image=data,
    method=Methods.NEAREST,
    group="/tmp/auto_multiscale2.zarr",

    # "axes" is an optional parameter, but only because write_image will just GUESS
    # the axis semantics for 2D images. Quite a human-facing decision
    axes=["c", "z", "y", "x"],
)




# Setting multiscale.images is invalid; `images` is a property and has no setter
#multiscales.images = []




# METADATA MANIPULATION #########################

# we can see the metada here, but it can't be modified, since it is
# frozen at the level of BaseAttrs
metadata_clone: MultiscaleV06 = multiscales.metadata.model_copy(deep=True)
try:
    metadata_clone.name = "blas" # will raise
except pydantic.ValidationError:
    pass

# we CAN set a new metadata, though:
multiscales.metadata = MultiscaleV06(
    name="another name",
    coordinateSystems=multiscales.metadata.coordinateSystems,
    # and we canc reate inconsistencies here too, so we end up with x images
    # but x-1 datasets in the metadata
    datasets=multiscales.metadata.datasets[1:],
    metadata=multiscales.metadata.metadata,
    type=multiscales.metadata.type,
)

#saving here just ignored the sabotaged datasets, probably only used the internal
# "images" as the source of truth
multiscales.to_ome_zarr("/tmp/auto_multiscale1.zarr")


# so I don't think modifying the emtadata is a good
# idea afterall, and if we wanted to change anything, we'd have to rebuild
# the entire Multiscales:


modified = OMEZarrMultiscale(
    image=multiscales.images[0],
    scale_factors=(4, 8)
    #...
)


# SCENES and transforms

# from the demo at come-zarr-py/.../create_scenes.ipynb:

from skimage import data

from ome_zarr import OMEZarrImage, OMEZarrMultiscale, OMEZarrScene

example_image = data.human_mitosis()

img1 = OMEZarrImage(data=example_image[:256, :256], axes=["y", "x"], name="img1")
img2 = OMEZarrImage(data=example_image[256:, :256], axes=["y", "x"], name="img2")
img3 = OMEZarrImage(data=example_image[:256, 256:], axes=["y", "x"], name="img3")
img4 = OMEZarrImage(data=example_image[256:, 256:], axes=["y", "x"], name="img4")

img1_ms = OMEZarrMultiscale(img1)
img2_ms = OMEZarrMultiscale(img2)
img3_ms = OMEZarrMultiscale(img3)
img4_ms = OMEZarrMultiscale(img4)

# the example on ome-zarr-py/.../create_scenes.ipynb notebook uses
# raw dicts for coord system and transforms, as they would be found in the spec,
# instead of instantiating proper classes. Still, the actual proper
# classes are allowed in the Scene constructor:

coordinate_system = CoordinateSystem(
    name="world",
    axes=(
        Axis(name="y", type="space"),
        Axis(name="x", type="space"),
    ),
)
coordinate_transformations = [
    Translation(
        translation=(0, 0),
        input=CoordinateSystemIdentifier(path="img1", name="physical"),
        output=CoordinateSystemIdentifier(name="world"),
    ),
    Translation(
        translation=(256, 0),
        input=CoordinateSystemIdentifier(path="img2", name="physical"),
        output=CoordinateSystemIdentifier(name="world"),
    ),
    Translation(
        translation=(0, 256),
        input=CoordinateSystemIdentifier(path="img3", name="physical"),
        output=CoordinateSystemIdentifier(name="world"),
    ),
    Translation(
        translation=(256, 256),
        input=CoordinateSystemIdentifier(path="img4", name="physical"),
        output=CoordinateSystemIdentifier(name="world"),
    ),
]

scene = OMEZarrScene(
    images=[img1_ms, img2_ms, img3_ms, img4_ms],
    coordinate_systems=[coordinate_system],
    coordinate_transformations=coordinate_transformations
)

# the example code shows the (transformnd) graph being accessed like so:
transf_seq: tnd.TransformSequence = scene._graph.get_sequence(
    source_space="img1:physical",
    target_space=":world"
)

# this kind of space: "img1:physical" is clearly a bug waiting to happen,
# and it should probably be at least a pair of strings to differentiate
# paths from names instead of encoding that with a colon in the string

# also... the graph is a private field. I suppose it will be exposed
# i different ways later, but right now, manipulating this graph
# would make the graph out of sync with the coord systems in the Scene.
# Plus, a strict type checker would complain on the access of the private
# variable.
