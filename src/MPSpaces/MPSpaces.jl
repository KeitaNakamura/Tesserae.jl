module MPSpaces

using Jams.Arrays: SparseArray, nonzeros, zeros!
using Jams.DofHelpers
using Jams.ShapeFunctions
using Jams.PointStates

import Jams.DofHelpers: ndofs
import Jams.ShapeFunctions: reinit!

using Base: @_propagate_inbounds_meta

export MPSpace

include("mpspace.jl")

end
