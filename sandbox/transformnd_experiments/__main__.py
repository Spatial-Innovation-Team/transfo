"""
Usage exemples/experiments of transformnd


"""

# pyright: strict

import time
import numpy as np
from transformnd.transforms import Displacements, Identity, Scale
import networkx.exception
print("importing is pretty slow, takes 50ms on my machine")
start = time.time()
from transformnd import Spaces, TransformGraph
print(f"Importing transformnd stuff tales {(time.time() - start) * 1000:.0f}ms")


# I'm using pyright to also check the ergonomics of the typing

# pyright: strict

graph = TransformGraph[np.ndarray]() # graphs are generic over the array type, but not over the space type?

#  I think I would have liked to see something like
#g = TransformGraph[np.ndarray, MySpatialdataGraphNode]
#
# I'm talking to Chris about this.
#
# Also, the fact that they are generic over the array type could be annoying for us
# if the SpatialData object is able to hold on to many kinds of array implementations.



A = "A"
B = "B"
C = "C"
D = "D"
E = "E"

scale_from_A_to_B = Scale(scale=[1,2], spaces=Spaces(A, B))


graph.add_transform(scale_from_A_to_B)
# no static check on this, though, but it raises an exception:
try:
    # this can take an `edge_data` dict, which weirdly exposes some internals, like a reserved key
    graph.add_transform(scale_from_A_to_B, source=C)
except ValueError:
    print("Detected bad source in runtime, as expected")

# can we add another edge from A to B, that completely messes up dimensions?
try:
    graph.add_transform(Scale(scale=[1,2,3,4,5,6], spaces=Spaces(A, B)))
except ValueError:
    pass # nope. error messag eis wonky, though


# we can add another edge from A to B. I don't know what criteria it uses to choose the path,
# but so far it seems like it's the first that wins out
graph.add_transform(Scale(scale=[100,200], spaces=Spaces(A, B)))
graph.add_transform(Scale(scale=[100,200], spaces=Spaces(A, B)))
graph.add_transform(Scale(scale=[100,200], spaces=Spaces(A, B)))
graph.add_transform(Scale(scale=[100,200], spaces=Spaces(A, B)))


from_a_to_b = graph.get_sequence(A, B)
points_in_a = np.asarray([
    [2,3],
    [4,5],
    [6,7],
    [8,9],
])
points_in_b = from_a_to_b.apply(points_in_a)

try:
    from_b_to_a = graph.get_sequence(B, A)
except networkx.exception.NetworkXNoPath:
    print("Raises on missing path, as expected.")
    print("     I don't know if I like that it's an 'internal' exception, from networkx")

try:
    from_a_to_missing_vert = graph.get_sequence(A, "i don't exist")
except networkx.exception.NodeNotFound:
    print("Raises, as expected")

try:
    from_b_to_a = graph.get_sequence(123, 456)
except networkx.exception.NodeNotFound:
    print("garbage verts are unfortunately indistiguishable form missing nodes here")


try:
    a_to_a = graph.get_sequence(A, A)
except ValueError:
    print(f"Weirdly, this raises a ValueError... feels like it should succeed with identity?")
    print(f"    Or at least fail with NetworkXNoPath? But I think networkx DID find a path; the empty one")

 #a bit odd that identity wants `ndim`. In spatial data we don't even bother with a constructor
ident = Identity(ndim=2)

# This looks a bit weird too (Identity that moves from one space to another) even though it's technically correct
ident2 = Identity(ndim=3, spaces=Spaces(B, C))


########################

displacement = Displacements(
    vector_field=np.asarray([
        [(+1, +2), (-1, +2)],
        [(-2, +3), (-5, +1)]
    ]),
    spaces=Spaces(D, E)

)
points_to_displace = np.asarray([
    [0, 1],
    [2, 1]
])
# interestingly, displacing [2,1] via `displacement` creates
# garbage values rather than an exception.
displaced_points = displacement.apply(points_to_displace)


##########



points_to_displace_f64 = np.asarray(
    [
        [0, 1],
        [2, 1]
    ],
    dtype=np.float64
)
displaced_f64 = displacement.apply(points_to_displace_f64)
# same garbage happens with floats too


########

# can we make disjoint subgraphs? Yes (displacement goes from D to E
graph.add_transform(displacement)

# it's a bit strange that the TransformGraph is aware of the number
# of dimensions in the spaces, but the spaces themselves aren't necessarily.
graph.space_ndims

# But since the space can be anything, we could implement that ourselves
class MyCoordSystem:
    name: str
    dimensions: ...








