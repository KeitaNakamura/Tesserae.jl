module MPSpaces

using Jams.Arrays:
# Collection
    AbstractCollection,
    UnionCollection,
    lazy

using Jams.TensorValues
using Jams.DofHelpers
using Jams.ShapeFunctions
using Jams.States

import Jams.DofHelpers: ndofs
import Jams.ShapeFunctions: reinit!
import Jams.States: pointstate, gridstate, set!

using Base: @_propagate_inbounds_meta

export MPSpace, ∑ᵢ, ∑ₚ, gridstate, pointstate, function_space, npoints

include("space.jl")
include("operations.jl")

end
