module Stencil

export
    AxisRegion,
    Boundary,
    Cell,
    Face,
    Ghost,
    Physical,
    Region,
    unitoffsets

include("offset.jl")
include("placement.jl")
include("region.jl")
include("indices.jl")

end # module Stencil
