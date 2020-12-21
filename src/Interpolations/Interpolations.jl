module Interpolations

using Reexport
@reexport using Tensors, Jams.Grids
import Tensors: gradient

using Base: @pure, @_propagate_inbounds_meta

export
# ShapeFunction
    ShapeFunction,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    BSplinePosition,
# Interpolation
    Interpolation,
    BSplineInterpolation,
    construct,
    nvalues,
    shape_value,
    shape_gradient,
    shape_symmetricgradient,
    reinit!


include("shape_function.jl")
include("interpolation.jl")
include("bsplines.jl")

end # module
