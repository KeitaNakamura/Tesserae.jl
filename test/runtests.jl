using Poingr
using Test

using Poingr: GridIndex

struct NodeState
    a::Float64
    b::Float64
end

include("maskedarray.jl")
include("grid.jl")

include("MaterialModels/SoilElastic.jl")
