module Stencil

import ..Tesserae: boundaries

export
    Cell,
    Face,
    Region,
    interior,
    boundaries,
    unitoffsets

include("location.jl")
include("offset.jl")
include("region.jl")
include("indices.jl")

end # Stencil
