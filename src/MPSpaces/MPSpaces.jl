module MPSpaces

using Jams.Arrays:
# SparseArray
    SparseArray,
    nonzeros,
    zeros!,
# Collection
    AbstractCollection,
    Collection,
    LazyCollection,
    changerank

using Jams.TensorValues
using Jams.DofHelpers
using Jams.ShapeFunctions
using Jams.PointStates

import Jams.TensorValues: valgrad
import Jams.DofHelpers: ndofs
import Jams.ShapeFunctions: reinit!
import Jams.PointStates: pointstate

using Base: @_propagate_inbounds_meta

export MPSpace, ∑ᵢ, ∑ₚ, gridstate, pointstate, npoints

include("sum.jl")
include("space.jl")

end
