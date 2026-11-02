module Stencil

import ..Tesserae: boundaries

export Cell, Face, Edge, Vertex, Region, interior, boundaries, unitoffsets, coordinates, @stencil

include("location.jl")
include("offset.jl")
include("region.jl")
include("indices.jl")
include("coordinates.jl")
include("broadcast.jl")
include("macro.jl")

end # Stencil
