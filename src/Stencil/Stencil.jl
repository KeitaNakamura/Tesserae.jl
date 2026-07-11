module Stencil

export
    AxisRegion,
    Boundary,
    Cell,
    Face,
    Halo,
    Physical,
    Region,
    mirror,
    unitoffsets

include("offset.jl")
include("placement.jl")
include("region.jl")
include("indices.jl")
include("mirror.jl")

end # module Stencil
