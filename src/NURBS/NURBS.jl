module NURBS

using Tensorial
import ..Tesserae: dropat, bspline_span_values!, greville_abscissae

include("utils.jl")
include("basis.jl")
include("controlnet.jl")
include("gmsh.jl")
include("primitives.jl")
include("degree_elevation.jl")
include("refinement.jl")
include("sweeps.jl")
include("coons_patch.jl")

end
