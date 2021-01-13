module DofHelpers

using Base: @_propagate_inbounds_meta

export
# DofMap
    DofMap,
    ndofs,
# DofMapIndices
    DofMapIndices,
    indices

include("DofMap.jl")
include("DofMapIndices.jl")

end
