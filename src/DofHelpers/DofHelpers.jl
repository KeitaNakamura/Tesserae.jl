module DofHelpers

using Base: @_propagate_inbounds_meta

export
    DofMap,
    DofMapIndices,
    ndofs,
    indices

include("dofmap.jl")

end
