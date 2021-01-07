module States

using Jams.Arrays: AbstractCollection, UnionCollection, LazyCollection, lazy
using Jams.DofHelpers
using Jams.Grids
using Base: @_propagate_inbounds_meta
using Base.Broadcast: broadcasted

import SparseArrays: nonzeros, nnz
import Jams.DofHelpers: indices

export
# PointState
    PointState,
    pointstate,
    ←,
    generate_pointstates,
# GridState
    GridState,
    gridstate,
    zeros!,
    nonzeros,
    nnz

include("pointstate.jl")
include("gridstate.jl")

end
