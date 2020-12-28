module DofHelpers

using Jams.Grids
using Base: @_propagate_inbounds_meta

export
# DofMap
    DofMap,
    DofMapIndices,
    ndofs,
    indices,
# PointToGridIndex
    PointToGridIndex,
    numbering!,
    dofindices,
    gridindices

include("dofmap.jl")
include("indexing.jl")

end
