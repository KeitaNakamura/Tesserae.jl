"""
    BSpline{order, dim}()
    LinearBSpline{dim}()
    QuadraticBSpline{dim}()
    CubicBSpline{dim}()

Create B-spline shape function.

# Examples
```jldoctest
julia> f = LinearBSpline{2}()
LinearBSpline{2}()

julia> f(Vec(0.5, 0.5))
0.25
```
"""
struct BSpline{order, dim} <: ShapeFunction{dim}
    function BSpline{order, dim}() where {order, dim}
        new{order::Int, dim::Int}()
    end
end

const LinearBSpline    = BSpline{1}
const QuadraticBSpline = BSpline{2}
const CubicBSpline     = BSpline{3}

support_length(::BSpline{1}) = 1.0
support_length(::BSpline{2}) = 1.5
support_length(::BSpline{3}) = 2.0
support_length(::BSpline{4}) = 2.5

function value(::BSpline{1, 0}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end

function value(::BSpline{2, 0}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end
@inline function value(::BSpline{2, 0}, ξ::Float64)
    ξ = abs(ξ)
    ξ < 0.5 ? 3/4 - ξ^2      :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end

function value(::BSpline{3, 0}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end
@inline function value(::BSpline{3, 0}, ξ::Float64)
    ξ = abs(ξ)
    ξ < 1 ? ξ^3/2 - ξ^2 + 2/3 :
    ξ < 2 ? (2 - ξ)^3 / 6     : zero(ξ)
end

function value(::BSpline{4, 0}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (48ξ^4 - 120ξ^2 + 115) / 192 :
    ξ < 1.5 ? -(16ξ^4 - 80ξ^3 + 120ξ^2 - 20ξ - 55) / 96 :
    ξ < 2.5 ? (5 - 2ξ)^4 / 384 : zero(ξ)
end

@inline function value(::BSpline{order, dim}, ξ::Vec{dim}) where {order, dim}
    prod(value.(BSpline{order, 0}(), ξ))
end

"""
    Poingr.BSplinePosition(; nth::Int, dir::Int)

Position of node at `ξ = 0`.
The node is located at `nth` away from bound with `dir` direction.

# Examples
```jldoctest
julia> Poingr.BSplinePosition([1,2,3,4,5], 2)
BSplinePosition(nth = 1, dir = 1)

julia> Poingr.BSplinePosition([1,2,3,4,5], 4)
BSplinePosition(nth = 1, dir = -1)

julia> pos = Poingr.BSplinePosition([1,2,3,4,5], 1)
BSplinePosition(nth = 0, dir = 0)

julia> f = QuadraticBSpline{0}()
QuadraticBSpline{0}()

julia> f(0.0, pos)
1.0
```
"""
struct BSplinePosition
    nth::Int
    dir::Int
end

BSplinePosition(; nth::Int, dir::Int) = BSplinePosition(nth, dir)

BSplinePosition(pos::Int) = BSplinePosition(abs(pos), sign(pos))

function BSplinePosition(v::AbstractVector, i::Int)
    @boundscheck checkbounds(v, i)
    l = i - firstindex(v)
    r = lastindex(v) - i
    pos = l < r ? l : -r
    BSplinePosition(pos)
end

function BSplinePosition(A::AbstractArray{<: Any, dim}, I::NTuple{dim, Int}) where {dim}
    @boundscheck checkbounds(A, I...)
    ntuple(Val(dim)) do d
        l = I[d] - firstindex(A, d)
        r = lastindex(A, d) - I[d]
        pos = l < r ? l : -r
        BSplinePosition(pos)
    end
end

BSplinePosition(A::AbstractArray{<: Any, dim}, I::CartesianIndex{dim}) where {dim} =
    (@_propagate_inbounds_meta; BSplinePosition(A, Tuple(I)))
BSplinePosition(A::AbstractArray{<: Any, dim}, I::Index{dim}) where {dim} =
    (@_propagate_inbounds_meta; BSplinePosition(A, I.I))

nthfrombound(pos::BSplinePosition) = pos.nth
dirfrombound(pos::BSplinePosition) = pos.dir

function value(spline::BSpline{1, 0}, ξ::Real, pos::BSplinePosition)::typeof(ξ)
    value(spline, ξ)
end

function value(spline::BSpline{2, 0}, ξ::Real, pos::BSplinePosition)::typeof(ξ)
    if nthfrombound(pos) == 0
        ξ = abs(ξ)
        ξ < 0.5 ? (3 - 4ξ^2) / 3 :
        ξ < 1.5 ? (3 - 2ξ)^2 / 6 : zero(ξ)
    elseif nthfrombound(pos) == 1
        ξ = dirfrombound(pos) * ξ
        ξ < -1   ? zero(ξ)                 :
        ξ < -0.5 ? 4(1 + ξ)^2 / 3          :
        ξ <  0.5 ? -(28ξ^2 - 4ξ - 17) / 24 :
        ξ <  1.5 ? (3 - 2ξ)^2 / 8          : zero(ξ)
    else
        value(spline, ξ)
    end
end

function value(spline::BSpline{3, 0}, ξ::Real, pos::BSplinePosition)::typeof(ξ)
    if nthfrombound(pos) == 0
        ξ = abs(ξ)
        ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 4 :
        ξ < 2 ? (2 - ξ)^3 / 4         : zero(ξ)
    elseif nthfrombound(pos) == 1
        ξ = dirfrombound(pos) * ξ
        ξ < -1 ? zero(ξ)                      :
        ξ <  0 ? (1 + ξ)^2 * (7 - 11ξ) / 12   :
        ξ <  1 ? (7ξ^3 - 15ξ^2 + 3ξ + 7) / 12 :
        ξ <  2 ? (2 - ξ)^3 / 6                : zero(ξ)
    else
        value(spline, ξ)
    end
end

@inline function value(::BSpline{order, dim}, ξ::Vec{dim}, pos::NTuple{dim, BSplinePosition}) where {order, dim}
    prod(value.(BSpline{order, 0}(), ξ, pos))
end

function Base.show(io::IO, pos::BSplinePosition)
    print(io, "BSplinePosition(nth = $(pos.nth), dir = $(pos.dir))")
end


struct BSplineValues{order, dim, T} <: ShapeValues{dim, T}
    F::BSpline{order, dim}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function BSplineValues{order, dim, T}() where {order, dim, T}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    BSplineValues(BSpline{order, dim}(), N, ∇N)
end

ShapeValues(::Type{T}, F::BSpline{order, dim}) where {order, dim, T} = BSplineValues{order, dim, T}()

function update!(it::BSplineValues{<: Any, dim}, grid, x::Vec{dim}, indices::AbstractArray = CartesianIndices(grid)) where {dim}
    @boundscheck checkbounds(grid, indices)
    F = it.F
    resize!(it.N, length(indices))
    resize!(it.∇N, length(indices))
    @inbounds @simd for i in 1:length(indices)
        I = indices[i]
        xᵢ = grid[I]
        it.N[i], it.∇N[i] = _value_gradient(F, x, xᵢ, gridsteps(grid), BSplinePosition(grid, I))
    end
    it
end

function _value(F::ShapeFunction, x::Vec{dim}, xᵢ::Vec{dim}, h::NTuple{dim}, pos) where {dim}
    ξ = (x - xᵢ) ./ h
    value(F, ξ, pos)
end

function _value_gradient(F::ShapeFunction, x::Vec, xᵢ::Vec, h::Tuple, pos)
    dv, v = gradient(x -> _value(F, x, xᵢ, h, pos), x, :all)
    v, dv
end


struct BSplineValue{dim, T}
    N::T
    ∇N::Vec{dim, T}
end

@inline function Base.getindex(it::BSplineValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(it.N[i], it.∇N[i])
end
