module States

using Poingr.Arrays
using Poingr.Collections
using Poingr.DofHelpers
using Poingr.Grids
using Base: @_propagate_inbounds_meta
using Base.Cartesian: @nall
using Base.Broadcast: broadcasted
using SparseArrays

import SparseArrays: nonzeros, nnz, sparse!, sparse
import Poingr.Arrays: nzindices
import Poingr.Collections: lazy, set!
import Poingr.ShapeFunctions: reinit!

export
# PointState
    PointState,
    pointstate,
    generate_pointstates,
# GridState
    GridState,
    GridStateThreads,
    gridstate,
    nonzeros, # from SparseArrays
    nnz,      # from SparseArrays
    zeros!,
    totalnorm,
# GridStateMatrix
    GridStateMatrix,
    gridstate_matrix,
    sparse,
    sparse!,
    solve!,
# GridDiagonal
    GridDiagonal,
# GridStateCollection
    GridStateCollection,
# PointToGridOperation
    ∑ₚ,
# GridToPointOperation
    ∑ᵢ,
# indices
    PointToDofIndices,
    PointToGridIndices

const PointToDofIndices = Vector{Vector{Int}}
const PointToGridIndices{dim} = Vector{Vector{CartesianIndex{dim}}}

include("PointState.jl")
include("GridState.jl")
include("GridStateMatrix.jl")
include("GridStateOperation.jl")
include("GridStateCollection.jl")
include("PointToGridOperation.jl")
include("GridToPointOperation.jl")
include("utils.jl")

end
