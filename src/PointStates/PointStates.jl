module PointStates

using Jams.Arrays: AbstractCollection, UnionCollection, LazyCollection, lazy
using Jams.Grids
using Base: @_propagate_inbounds_meta
using Base.Broadcast: broadcasted

export
# PointState
    PointState,
    pointstate,
    ←,
# generation
    generate_pointstates

include("pointstate.jl")
include("generation.jl")

end
