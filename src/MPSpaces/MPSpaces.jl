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
import Jams.TensorValues: valgrad

using Jams.DofHelpers
using Jams.ShapeFunctions
using Jams.PointStates

import Jams.DofHelpers: ndofs
import Jams.ShapeFunctions: reinit!

using Base: @_propagate_inbounds_meta

export MPSpace, ∑ᵢ, ∑ₚ

include("sum.jl")
include("space.jl")

end
