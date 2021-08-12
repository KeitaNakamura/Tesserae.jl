using Poingr
using Test

struct NodeState
    a::Float64
    b::Float64
end

include("maskedarray.jl")

include("MaterialModels/SoilElastic.jl")
