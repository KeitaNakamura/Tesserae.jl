module Stencil

export
    AxisRegion,
    Boundary,
    Cell,
    Edge,
    Face,
    Full,
    Halo,
    Physical,
    Region,
    Vertex,
    reflect,
    unitoffsets

include("offset.jl")
include("placement.jl")
include("region.jl")
include("indices.jl")
include("reflection.jl")

end # module Stencil
