module MPSpaces

using Jams.Arrays:
# SparseArray
    SparseArray,
    nonzeros,
    zeros!,
# Collection
    UnionCollection

using Jams.TensorValues
using Jams.DofHelpers
using Jams.ShapeFunctions
using Jams.PointStates

import Jams.TensorValues: valgrad
import Jams.DofHelpers: ndofs
import Jams.ShapeFunctions: reinit!
import Jams.PointStates: pointstate, set!

using Base: @_propagate_inbounds_meta

export MPSpace, ∑ᵢ, ∑ₚ, gridstate, pointstate, npoints

include("space.jl")
include("operations.jl")

end
