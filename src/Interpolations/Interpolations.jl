module Interpolations

using Reexport
@reexport using Tensors, Jams.Grids

using Jams.TensorValues
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
    VectorValue,
    BSplineInterpolation,
    construct,
    reinit!


include("shape_function.jl")
include("interpolation.jl")
include("bsplines.jl")

end # module
