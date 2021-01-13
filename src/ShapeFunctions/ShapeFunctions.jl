module ShapeFunctions

using Reexport
@reexport using Jams.TensorValues, Jams.Grids
using Jams.Collections

import Jams.TensorValues: gradient

using Base: @pure, @_propagate_inbounds_meta

export
# ShapeFunction
    ShapeFunction,
# BSpline
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    BSplinePosition,
# ShapeValue
    ShapeValue,
    construct,
    reinit!,
# BSplineValue
    BSplineValue

include("ShapeFunction.jl")
include("BSpline.jl")

include("ShapeValue.jl")
include("BSplineValue.jl")

end # module
