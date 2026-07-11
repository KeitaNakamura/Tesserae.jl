module Stencil

export
    AxisRegion,
    Boundary,
    Cell,
    Face,
    Halo,
    Physical,
    Region,
    reflect,
    unitoffsets

include("offset.jl")
include("placement.jl")
include("region.jl")
include("indices.jl")
include("reflection.jl")

end # module Stencil
