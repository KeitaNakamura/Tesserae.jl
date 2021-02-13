module States

using Jams.Arrays
using Jams.Collections
using Jams.DofHelpers
using Jams.Grids
using Base: @_propagate_inbounds_meta
using Base.Cartesian: @nall
using Base.Broadcast: broadcasted
using SparseArrays

import SparseArrays: nonzeros, nnz, sparse!, sparse
import Jams.Collections: lazy, set!
import Jams.DofHelpers: indices
import Jams.ShapeFunctions: reinit!

export
# PointState
    PointState,
    pointstate,
    ←,
    generate_pointstates,
# GridState
    GridState,
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
    ∑ᵢ

include("PointState.jl")
include("GridState.jl")
include("GridStateMatrix.jl")
include("GridStateOperation.jl")
include("GridStateCollection.jl")
include("PointToGridOperation.jl")
include("GridToPointOperation.jl")
include("utils.jl")

end
