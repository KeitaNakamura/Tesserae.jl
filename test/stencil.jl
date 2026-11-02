using Tesserae.Stencil

module HeatExample
const RUN_TESTS = true
include("../docs/literate/tutorials/finite_volume.jl")
end

module ReferenceHeat
include("stencil/reference_heat.jl")
end

@testset "Staggered-grid stencils" begin
    include("stencil/regions.jl")
    include("stencil/coordinates.jl")
    include("stencil/offsets.jl")
    include("stencil/conservation.jl")
    include("stencil/broadcast.jl")
    include("stencil/macro.jl")
    include("stencil/mac.jl")
end
