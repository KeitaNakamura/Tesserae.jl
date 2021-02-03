module MPSpaces

using Jams.Arrays
using Jams.TensorValues
using Jams.DofHelpers
using Jams.ShapeFunctions
using Jams.States
using Jams.Collections: lazy

import Jams.DofHelpers: ndofs
import Jams.ShapeFunctions: reinit!, construct
import Jams.States: pointstate, gridstate, gridstate_matrix, set!

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
