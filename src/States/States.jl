module States

using Jams.Arrays: AbstractCollection, LazyCollection, Collection, lazy, SparseMatrixCOO
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
    GridCollection,
    GridStateMatrix,
    gridstate,
    gridstate_matrix,
    zeros!,
    nonzeros,
    nnz

include("pointstate.jl")
include("gridstate.jl")

end
