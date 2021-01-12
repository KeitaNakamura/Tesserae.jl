module MPSpaces

using Jams.Arrays:
# Collection
    AbstractCollection,
    lazy,
# FillArray
    FillArray,
    Ones,
    Zeros

using Jams.TensorValues
using Jams.DofHelpers
using Jams.ShapeFunctions
using Jams.States

import Jams.DofHelpers: ndofs
import Jams.ShapeFunctions: reinit!
import Jams.States: pointstate, gridstate, gridstate_matrix, set!

using Base: @_propagate_inbounds_meta

export
    MPSpace,
    gridstate,
    gridstate_matrix,
    pointstate,
    function_space,
    npoints,
    dirichlet!,
    ∑ᵢ,
    ∑ₚ

include("space.jl")
include("operations.jl")

end
