"""
    BSpline{order}(dim)
    BSpline{order}(; dim)

Create B-spline shape function.

# Examples
```jldoctest
julia> f = BSpline{1}(dim = 2)
BSpline{1}(dim = 2)

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

@pure BSpline{order}(dim::Int) where {order} = BSpline{order, dim}()
@pure BSpline{order}(; dim::Int) where {order} = BSpline{order, dim}()

support_length(::BSpline{1}) = 1.0
support_length(::BSpline{2}) = 1.5
support_length(::BSpline{3}) = 2.0
support_length(::BSpline{4}) = 2.5

function value(::BSpline{1, 0}, ξ::Real)::typeof(ξ)
    ξ = abs(ξ)
    iszero(ξ) && return one(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end

function value(::BSpline{2, 0}, ξ::Real)::typeof(ξ)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end

function value(::BSpline{3, 0}, ξ::Real)::typeof(ξ)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end

function value(::BSpline{4, 0}, ξ::Real)::typeof(ξ)
    ξ = abs(ξ)
    ξ < 0.5 ? (48ξ^4 - 120ξ^2 + 115) / 192 :
    ξ < 1.5 ? -(16ξ^4 - 80ξ^3 + 120ξ^2 - 20ξ - 55) / 96 :
    ξ < 2.5 ? (5 - 2ξ)^4 / 384 : zero(ξ)
end

function value(::BSpline{order, dim}, ξ::Vec{dim}) where {order, dim}
    prod(i -> value(BSpline{order}(dim = 0), @inbounds(ξ[i])), 1:dim)
end

function Base.show(io::IO, x::BSpline{order, dim}) where {order, dim}
    print(io, "BSpline{$order}(dim = $dim)")
end


"""
    BSplinePosition(; nth::Int, dir::Int)

Position of node at `ξ = 0`.
The node is located at `nth` away from bound with `dir` direction.

# Examples
```jldoctest
julia> BSplinePosition([1,2,3,4,5], 2)
BSplinePosition(nth = 1, dir = 1)

julia> BSplinePosition([1,2,3,4,5], 4)
BSplinePosition(nth = 1, dir = -1)

julia> pos = BSplinePosition([1,2,3,4,5], 1)
BSplinePosition(nth = 0, dir = 0)

julia> f = BSpline{2}(dim = 0)
BSpline{2}(dim = 0)

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
        l = I[d] - firstindex(A, I[d])
        r = lastindex(A, I[d]) - I[d]
        pos = l < r ? l : -r
        BSplinePosition(pos)
    end
end

BSplinePosition(A::AbstractArray{<: Any, dim}, I::CartesianIndex{dim}) where {dim} =
    (@_propagate_inbounds_meta; BSplinePosition(A, Tuple(I)))

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

function value(::BSpline{3, 0}, ξ::Real, pos::BSplinePosition)::typeof(ξ)
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

function value(::BSpline{order, dim}, ξ::Vec{dim}, pos::NTuple{dim, BSplinePosition}) where {order, dim}
    prod(i -> value(BSpline{order}(dim = 0), @inbounds(ξ[i]), @inbounds(pos[i])), 1:dim)
end

function Base.show(io::IO, pos::BSplinePosition)
    print(io, "BSplinePosition(nth = $(pos.nth), dir = $(pos.dir))")
end


struct BSplineValue{order, dim, T} <: ShapeValue{dim, T}
    F::BSpline{order, dim}
    N::Vector{T}
    dN::Vector{Vec{dim, T}}
end

function construct(::Type{T}, F::BSpline{order, dim}) where {order, dim, T}
    N = Vector{T}(undef, 0)
    dN = Vector{Vec{dim, T}}(undef, 0)
    BSplineValue(F, N, dN)
end

function reinit!(it::BSplineValue{<: Any, dim}, grid::AbstractGrid{dim}, indices::AbstractArray, x::Vec{dim}) where {dim}
    @boundscheck checkbounds(grid, indices)
    F = it.F
    resize!(it.N, length(indices))
    resize!(it.dN, length(indices))
    j = 1
    @inbounds for I in view(CartesianIndices(grid), indices)
        xᵢ = grid[I]
        ξ = Vec{dim}(i -> @inbounds((x[i] - xᵢ[i]) / gridsteps(grid, i)))
        it.N[j], it.dN[j] = value_gradient(F, ξ, BSplinePosition(grid, I))
        j += 1
    end
    it
end
