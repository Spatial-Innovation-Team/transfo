# pyright: strict

from typing import cast

import dask.array as da
from dask.array.core import Array as DaskArray


from ngff_zarr.methods import Methods
from ngff_zarr.to_ngff_zarr import to_multiscales
import ngff_zarr.v06.zarr_metadata as v6meta
from ngff_zarr.multiscales import NgffMultiscales


# you can instantiate a standalone image:
from ngff_zarr.ngff_image import NgffImage
img_data: DaskArray = cast(DaskArray, da.from_array([1,2,3])) # pyright: ignore
img = NgffImage(
    data=img_data, # must always a DaskArray
    dims=["x", "y"],
    scale={
        "x": 1.0,
        "y": 2.0,
    },
    translation={
        "x": 0.0,
        "y": 0.0,
    },
)

# NgffImage is a @dataclass, so there is no validation on dims/scale/translation.
# It also exposes some rather internal stuff, like the .computed_callbacks field.
#
# Curiously, there is no ngff metadata inside the NgffImage, nor methods on it
# to generate the v6meta.Dataset. All metadata is found one level higher,
# in the NgffMultiscales class:
multiscales = NgffMultiscales(
    images=[img],
    metadata=v6meta.Metadata(
        coordinateSystems=[],
        coordinateTransformations=[],
        datasets=[
            # this would be the metadata for `img: NgffImage`, and is also
            # surprisingly detached from NgffImage itself
            v6meta.Dataset(
                path="bla/ble",
                coordinateTransformations=[],
            )
        ]
    )
)

# then NgffMultiscales can save to disk via
multiscales.to_ome_zarr("/tmp/ngff_zarr__bla.zarr") # pyright: ignore

# just like ome-zarr-py, they also expose **kwargs that are
# forwarded to zarr.create_array, leaking implementation details downstream.

##############

# you can also generate the downscaled levels like so:
generated_multiscales = to_multiscales(
    data=img,
    # i think this means: "keeps downscaling until the tiles have size of 64"?
    #
    # I would have liked to see more explicit typing of what is allowed here, something
    # like "SmallestTargetDimensions | ScaleFactorPerDownscaling" or something equivalent
    scale_factors=64,
    method=Methods.DASK_IMAGE_GAUSSIAN,
)

#

### SCENES #######3

# I have only found json schemas for Scenes in ngff_zarr, but no classes.
