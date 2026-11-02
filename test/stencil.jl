using Tesserae, Tesserae.Stencil, Test, LinearAlgebra, Random

@testset "Staggered-grid stencils" begin
    include("stencil/backend.jl")
    include("stencil/regions.jl")
    include("stencil/coordinates.jl")
    include("stencil/offsets.jl")
    include("stencil/conservation.jl")
    include("stencil/broadcast.jl")
    include("stencil/macro.jl")
    include("stencil/heat.jl")
    include("stencil/mac.jl")
end
