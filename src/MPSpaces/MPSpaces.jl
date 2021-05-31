module MPSpaces

using Poingr.Arrays
using Poingr.TensorValues
using Poingr.DofHelpers
using Poingr.ShapeFunctions
using Poingr.States
using Poingr.Collections

import Poingr.DofHelpers: ndofs
import Poingr.ShapeFunctions: reinit!, construct
import Poingr.States: pointstate, gridstate, gridstate_matrix, set!

using Base: @_propagate_inbounds_meta

export
    MPSpace,
    pointstate,
    gridstate,
    gridstate_matrix,
    function_space,
    npoints,
    dirichlet!,
    boundary,
    nearsurface

include("BoundNormalArray.jl")
include("MPSpace.jl")
include("MPSpaceBound.jl")
include("MPSpaceNearSurface.jl")

end
