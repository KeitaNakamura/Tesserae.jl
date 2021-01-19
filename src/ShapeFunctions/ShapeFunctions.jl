module ShapeFunctions

using Reexport
@reexport using Jams.TensorValues, Jams.Grids
using Jams.Collections
using StaticArrays

import Jams.TensorValues: gradient, ∇

using Base: @pure, @_propagate_inbounds_meta

export
# ShapeFunction
    ShapeFunction,
# ShapeValue
    ShapeValue,
    construct,
    reinit!,
# BSpline
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    BSplinePosition,
# BSplineValue
    BSplineValue,
# Polynomial
    Polynomial,
    polynomial,
# WLS
    WLS,
# WLSValue
    WLSValue

include("ShapeFunction.jl")
include("ShapeValue.jl")
include("BSpline.jl")
include("BSplineValue.jl")
include("Polynomial.jl")
include("WLS.jl")
include("WLSValue.jl")

end # module
