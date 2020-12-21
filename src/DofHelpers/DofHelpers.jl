module DofHelpers

using Jams.Grids
using Base: @_propagate_inbounds_meta

export
# DofMap
    DofMap,
    numbering!,
    ndofs,
# PointToGridIndex
    PointToGridIndex,
    dofindices,
    gridindices

include("dofmap.jl")
include("indexing.jl")

end
