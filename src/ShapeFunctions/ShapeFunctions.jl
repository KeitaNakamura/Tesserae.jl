module ShapeFunctions

using Reexport
@reexport using Tensors, Jams.Grids

using Jams.Arrays: AbstractCollection, LazyCollection, changerank
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
# ShapeValue
    ShapeValue,
    BSplineValue,
    construct,
    reinit!,
    ∑ₚ,
    ∑ᵢ


include("shape_function.jl")
include("shape_value.jl")
include("bsplines.jl")

end # module
