using Poingr
using Test

using Poingr: Index

struct NodeState
    a::Float64
    b::Float64
end

include("utils.jl")
include("maskedarray.jl")
include("grid.jl")

include("MaterialModels/SoilElastic.jl")
