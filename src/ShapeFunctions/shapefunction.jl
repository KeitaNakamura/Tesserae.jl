abstract type ShapeFunction{dim} end

Broadcast.broadcastable(f::ShapeFunction) = (f,)

# scalar
(f::ShapeFunction{0})(ξ::Real, args...) where {dim} = value(f, ξ, args...)

function value_derivative(f::ShapeFunction, ξ::Real, args...)
    res = DiffResults.DiffResult(zero(ξ), zero(ξ))
    res = ForwardDiff.derivative!(res, ξ -> value(f, ξ, args...), ξ)
    DiffResults.value(res), DiffResults.derivative(res)
end

function derivative(f::ShapeFunction, ξ::Real, args...)
    value_derivative(f, ξ, args...)[2]
end

# vector
(f::ShapeFunction{dim})(ξ::Vec{dim}, args...) where {dim} = value(f, ξ, args...)

function value_gradient(f::ShapeFunction{dim}, ξ::Vec{dim}, args...) where {dim}
    dv, v = gradient(ξ -> value(f, ξ, args...), ξ, :all)
    v, dv
end

function Tensorial.gradient(f::ShapeFunction{dim}, ξ::Vec{dim}, args...) where {dim}
    value_gradient(f, ξ, args...)[2]
end


# TODO: fix type instablity when `f''(ξ)`
struct GradientShapeFunction{dim, F <: ShapeFunction{dim}} <: ShapeFunction{dim}
    f::F
end

Base.adjoint(f::ShapeFunction) = GradientShapeFunction(f)
# TensorValues.∇(f::ShapeFunction) = GradientShapeFunction(f)

value(∇f::GradientShapeFunction{0}, ξ::Real, args...) = derivative(∇f.f, ξ, args...)
value(∇f::GradientShapeFunction{dim}, ξ::Vec{dim}, args...) where {dim} = gradient(∇f.f, ξ, args...)



abstract type ShapeValues{dim, T} <: AbstractVector{T} end

Base.size(x::ShapeValues) = size(x.N)

"""
    Poingr.ShapeValues(::ShapeFunction)
    Poingr.ShapeValues(::Type{T}, ::ShapeFunction)

Construct object storing value of `ShapeFunction`.

# Examples
```jldoctest
julia> sv = Poingr.ShapeValues(QuadraticBSpline{2}());

julia> reinit!(sv, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(sv.N)
1.0

julia> sum(sv.∇N)
2-element Vec{2, Float64}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
ShapeValues(F::ShapeFunction{dim}) where {dim} = ShapeValues(Float64, F)

"""
    reinit!(::ShapeValues, grid::Grid, x::Vec)
    reinit!(::ShapeValues, grid::Grid, indices::AbstractArray, x::Vec)

Reinitialize value of shape function at `x` with each `grid` node.

# Examples
```jldoctest
julia> sv = Poingr.ShapeValues(QuadraticBSpline{2}());

julia> reinit!(sv, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(sv.N)
1.0

julia> reinit!(sv, Grid(0:3, 0:3), Vec(1, 1), CartesianIndices((1:2, 1:2)));

julia> sum(sv.N)
0.765625
```
"""
reinit!

function reinit!(it::ShapeValues, grid::Grid, x::Vec)
    reinit!(it, grid, eachindex(grid), x)
end

Base.length(it::ShapeValues) = length(it.N)
