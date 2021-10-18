using Poingr
using Random
using Test

using Poingr: Index

struct NodeState
    a::Float64
    b::Float64
end

include("utils.jl")
include("sparray.jl")
include("grid.jl")
include("shapefunctions.jl")

include("mpcache.jl")

include("MaterialModels/SoilHyperelastic.jl")
include("MaterialModels/DruckerPrager.jl")

@testset "Run examples" begin
    include("../examples/sandcolumn.jl")
    SandColumn.main(; shape_function = QuadraticBSpline(), show_progress = false)
    SandColumn.main(; shape_function = LinearWLS(QuadraticBSpline()), show_progress = false)
end
