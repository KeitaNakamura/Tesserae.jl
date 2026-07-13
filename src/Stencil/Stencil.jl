module Stencil

export
    AxisRegion,
    Cell,
    Edge,
    Face,
    Region,
    Vertex,
    full,
    physical,
    lowhalo,
    highhalo,
    lowboundary,
    highboundary,
    reflect,
    unitoffsets

include("offset.jl")
include("location.jl")
include("region.jl")
include("indices.jl")
include("reflection.jl")

end # module Stencil
