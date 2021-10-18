"""
    BSpline{order}()
    LinearBSpline()
    QuadraticBSpline()
    CubicBSpline()

Create B-spline shape function.

# Examples
```jldoctest
julia> f = LinearBSpline()
LinearBSpline()

julia> Poingr.value(f, Vec(0.5, 0.5))
0.25
```
"""
struct BSpline{order} <: ShapeFunction
    function BSpline{order}() where {order}
        new{order::Int}()
    end
end

const LinearBSpline    = BSpline{1}
const QuadraticBSpline = BSpline{2}
const CubicBSpline     = BSpline{3}

support_length(::BSpline{1}) = 1.0
support_length(::BSpline{2}) = 1.5
support_length(::BSpline{3}) = 2.0
support_length(::BSpline{4}) = 2.5
active_length(bspline::BSpline) = support_length(bspline) # for sparsity pattern

@pure nnodes(bspline::BSpline, ::Val{dim}) where {dim} = prod(nfill(Int(2*support_length(bspline)), Val(dim)))

function value(::BSpline{1}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end

function value(::BSpline{2}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end
@inline function value(::BSpline{2}, ξ::Float64)
    ξ = abs(ξ)
    ξ < 0.5 ? 0.75 - ξ^2         :
    ξ < 1.5 ? 0.125 * (3 - 2ξ)^2 : zero(ξ)
end

function value(::BSpline{3}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end
@inline function value(::BSpline{3}, ξ::Float64)
    ξ = abs(ξ)
    ξ < 1 ? ξ^3/2 - ξ^2 + 2/3 :
    ξ < 2 ? (2 - ξ)^3 / 6     : zero(ξ)
end

function value(::BSpline{4}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (48ξ^4 - 120ξ^2 + 115) / 192 :
    ξ < 1.5 ? -(16ξ^4 - 80ξ^3 + 120ξ^2 - 20ξ - 55) / 96 :
    ξ < 2.5 ? (5 - 2ξ)^4 / 384 : zero(ξ)
end

@generated function value(bspline::BSpline, ξ::Vec{dim}) where {dim}
    exps = [:(value(bspline, ξ[$i])) for i in 1:dim]
    quote
        @_inline_meta
        *($(exps...))
    end
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

julia> f = QuadraticBSpline()
QuadraticBSpline()

julia> Poingr.value(f, 0.0, pos)
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

function value(spline::BSpline{1}, ξ::Real, pos::BSplinePosition)::typeof(ξ)
    value(spline, ξ)
end

function value(spline::BSpline{2}, ξ::Real, pos::BSplinePosition)::typeof(ξ)
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

function value(spline::BSpline{3}, ξ::Real, pos::BSplinePosition)::typeof(ξ)
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

@inline function value(bspline::BSpline, ξ::Vec{dim}, pos::NTuple{dim, BSplinePosition}) where {dim}
    prod(value.(Ref(bspline), ξ, pos))
end

function Base.show(io::IO, pos::BSplinePosition)
    print(io, "BSplinePosition(nth = $(pos.nth), dir = $(pos.dir))")
end


struct BSplineValues{order, dim, T, L} <: ShapeValues{dim, T}
    F::BSpline{order}
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    inds::MVector{L, Index{dim}}
    len::Base.RefValue{Int}
end

function BSplineValues{order, dim, T, L}() where {order, dim, T, L}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    inds = MVector{L, Index{dim}}(undef)
    BSplineValues(BSpline{order}(), N, ∇N, inds, Ref(0))
end

function ShapeValues{dim, T}(F::BSpline{order}) where {order, dim, T}
    L = nnodes(F, Val(dim))
    BSplineValues{order, dim, T, L}()
end

function update!(it::BSplineValues{<: Any, dim}, grid, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    update_gridindices!(it, grid, x, spat)
    it.N .= zero(it.N)
    it.∇N .= zero(it.∇N)
    F = it.F
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        it.∇N[i], it.N[i] = gradient(x, :all) do x
            @_inline_meta
            ξ = (x - xᵢ) ./ gridsteps(grid)
            value(F, ξ, BSplinePosition(grid, I))
        end
    end
    it
end

struct BSplineValue{dim, T}
    N::T
    ∇N::Vec{dim, T}
    index::Index{dim}
end

@inline function Base.getindex(it::BSplineValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(it.N[i], it.∇N[i], it.inds[i])
end
