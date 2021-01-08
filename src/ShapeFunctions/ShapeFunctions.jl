module ShapeFunctions

using Reexport
@reexport using Jams.TensorValues, Jams.Grids

using Jams.Arrays: AbstractCollection
import Jams.TensorValues: gradient

using Base: @pure, @_propagate_inbounds_meta

export
# ShapeFunction
    ShapeFunction,
    BSpline,
    LinearBSpline,
    QuadraticBSpline,
    CubicBSpline,
    BSplinePosition,
# ShapeValue
    ShapeValue,
    BSplineValue,
    construct,
    reinit!


include("shape_function.jl")
include("shape_value.jl")
include("bsplines.jl")

end # module
