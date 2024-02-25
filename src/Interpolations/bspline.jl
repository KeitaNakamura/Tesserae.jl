struct BSpline{order} <: Kernel
    BSpline{order}() where {order} = new{order::Int}()
end

"""
    BSpline{1}()
    LinearBSpline()

Linear B-spline kernel.
"""
const LinearBSpline = BSpline{1}

"""
    BSpline{2}()
    QuadraticBSpline()

Quadratic B-spline kernel.
The peaks of this funciton are centered on the grid nodes [^Steffen].

[^Steffen]: [Steffen, M., Kirby, R. M., & Berzins, M. (2008). Analysis and reduction of quadrature errors in the material point method (MPM). *International journal for numerical methods in engineering*, 76(6), 922-948.](https://doi.org/10.1002/nme.2360)
"""
const QuadraticBSpline = BSpline{2}

"""
    BSpline{3}()
    CubicBSpline()

Cubic B-spline kernel.
The peaks of this funciton are centered on the grid nodes [^Steffen].

[^Steffen]: [Steffen, M., Kirby, R. M., & Berzins, M. (2008). Analysis and reduction of quadrature errors in the material point method (MPM). *International journal for numerical methods in engineering*, 76(6), 922-948.](https://doi.org/10.1002/nme.2360)
"""
const CubicBSpline = BSpline{3}

gridspan(::BSpline{1}) = 2
gridspan(::BSpline{2}) = 3
gridspan(::BSpline{3}) = 4

@inline function neighbornodes(bs::BSpline, pt, lattice::Lattice{dim, T}) where {dim, T}
    x = getx(pt)
    isinside(x, lattice) || return CartesianIndices(nfill(1:0, Val(dim)))
    _neighbornodes(bs, SVec{dim,T}(x), SVec{dim,Int}(size(lattice)), spacing_inv(lattice), SVec{dim,T}(first(lattice)))
end
@inline function _neighbornodes(::BSpline{1}, x::SVec{dim, T}, dims::SVec{dim, Int}, dx⁻¹::T, xmin::SVec{dim, T}) where {dim, T}
    _bspline_neighborindices(x, 1, T(0), dims, dx⁻¹, xmin)
end
@inline function _neighbornodes(::BSpline{2}, x::SVec{dim, T}, dims::SVec{dim, Int}, dx⁻¹::T, xmin::SVec{dim, T}) where {dim, T}
    _bspline_neighborindices(x, 2, T(0.5), dims, dx⁻¹, xmin)
end
@inline function _neighbornodes(::BSpline{3}, x::SVec{dim, T}, dims::SVec{dim, Int}, dx⁻¹::T, xmin::SVec{dim, T}) where {dim, T}
    _bspline_neighborindices(x, 3, T(1), dims, dx⁻¹, xmin)
end
@inline function _bspline_neighborindices(x::SVec{dim, T}, h::Int, offset::T, dims::SVec{dim, Int}, dx⁻¹::T, xmin::SVec{dim, T}) where {dim, T}
    ξ = (x - xmin) * dx⁻¹
    start = convert(SVec{dim, Int}, floor(ξ - offset)) + 1
    stop = start + h
    imin = Tuple(max(start, 1))
    imax = Tuple(min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end

# simple B-spline calculations
function value(::BSpline{1}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end
function value(::BSpline{2}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end
function value(::BSpline{3}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end
@generated function value(bspline::BSpline, ξ::Vec{dim}) where {dim}
    quote
        @_inline_meta
        prod(@ntuple $dim i -> value(bspline, ξ[i]))
    end
end
@inline function value(bspline::BSpline, xₚ::Vec, lattice::Lattice, I::CartesianIndex)
    @_propagate_inbounds_meta
    xᵢ = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xₚ - xᵢ) * dx⁻¹
    value(bspline, ξ)
end
@inline function value(bspline::BSpline, pt, lattice::Lattice, I::CartesianIndex)
    @_propagate_inbounds_meta
    value(bspline, getx(pt), lattice, I)
end

# Steffen, M., Kirby, R. M., & Berzins, M. (2008).
# Analysis and reduction of quadrature errors in the material point method (MPM).
# International journal for numerical methods in engineering, 76(6), 922-948.
function value(spline::BSpline{1}, ξ::Real, pos::Int)::typeof(ξ)
    value(spline, ξ)
end
function value(spline::BSpline{2}, ξ::Real, pos::Int)::typeof(ξ)
    if pos == 0
        ξ = abs(ξ)
        ξ < 0.5 ? (3 - 4ξ^2) / 3 :
        ξ < 1.5 ? (3 - 2ξ)^2 / 6 : zero(ξ)
    elseif abs(pos) == 1
        ξ = sign(pos) * ξ
        ξ < -1   ? zero(ξ)                 :
        ξ < -0.5 ? 4(1 + ξ)^2 / 3          :
        ξ <  0.5 ? -(28ξ^2 - 4ξ - 17) / 24 :
        ξ <  1.5 ? (3 - 2ξ)^2 / 8          : zero(ξ)
    else
        value(spline, ξ)
    end
end
function value(spline::BSpline{3}, ξ::Real, pos::Int)::typeof(ξ)
    if pos == 0
        ξ = abs(ξ)
        ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 4 :
        ξ < 2 ? (2 - ξ)^3 / 4         : zero(ξ)
    elseif abs(pos) == 1
        ξ = sign(pos) * ξ
        ξ < -1 ? zero(ξ)                      :
        ξ <  0 ? (1 + ξ)^2 * (7 - 11ξ) / 12   :
        ξ <  1 ? (7ξ^3 - 15ξ^2 + 3ξ + 7) / 12 :
        ξ <  2 ? (2 - ξ)^3 / 6                : zero(ξ)
    else
        value(spline, ξ)
    end
end
@generated function value(bspline::BSpline, ξ::Vec{dim}, pos::Tuple{Vararg{Int, dim}}) where {dim}
    quote
        @_inline_meta
        prod(@ntuple $dim i -> value(bspline, ξ[i], pos[i]))
    end
end
@inline function value(bspline::BSpline, xₚ::Vec, lattice::Lattice, I::CartesianIndex, ::Symbol) # last argument is pseudo argument `:steffen`
    @_propagate_inbounds_meta
    xᵢ = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xₚ - xᵢ) * dx⁻¹
    value(bspline, ξ, node_position(lattice, I))
end

@inline function node_position(ax::AbstractVector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
node_position(lattice::Lattice, index::CartesianIndex) = map(node_position, get_axes(lattice), Tuple(index))


@inline fract(x) = x - floor(x)
# Fast calculations for values and gradients
# `x` must be normalized by `dx`
@inline function Base.values(::BSpline{1}, x::Real, g=nothing)
    T = typeof(x)
    V = NTuple{2, T}
    ξ = fract(x)
    vals = V((1-ξ, ξ))
    if g isa Symbol
        vals, V((-1, 1))
    else
        vals
    end
end
@inline function Base.values(::BSpline{2}, x::Real, g=nothing)
    T = typeof(x)
    V = NTuple{3, T}
    x′ = fract(x - T(0.5))
    ξ = @. x′ - $V((-0.5,0.5,1.5))
    vals = @. $V((0.5,-1.0,0.5))*ξ^2 + $V((-1.5,0.0,1.5))*ξ + $V((1.125,0.75,1.125))
    if g isa Symbol
        vals, @. $V((1.0,-2.0,1.0))*ξ + $V((-1.5,0.0,1.5))
    else
        vals
    end
end
@inline function Base.values(::BSpline{3}, x::Real, g=nothing)
    T = typeof(x)
    V = NTuple{4, T}
    x′ = fract(x)
    ξ = @. x′ - $V((-1,0,1,2))
    ξ² = @. ξ * ξ
    ξ³ = @. ξ² * ξ
    vals = @. $V((-1/6,0.5,-0.5,1/6))*ξ³ + $V((1,-1,-1,1))*ξ² + $V((-2,0,0,2))*ξ + $V((4/3,2/3,2/3,4/3))
    if g isa Symbol
        vals, @. $V((-0.5,1.5,-1.5,0.5))*ξ² + $V((2,-2,-2,2))*ξ + $V((-2,0,0,2))
    else
        vals
    end
end

@generated function Base.values(bspline::BSpline, x::Vec{dim}, ::Symbol) where {dim}
    quote
        @_inline_meta
        @nexprs $dim d -> (x_d, ∇x_d) = values(bspline, x[d], :withgradient)
        vals = @ntuple $dim d -> x_d
        valsgrads = @ntuple $dim d -> (@ntuple $dim j -> j==d ? ∇x_j : x_j)
        otimes_tuple(vals), map(Vec, map(otimes_tuple, valsgrads)...)
    end
end
@inline function Base.values(bspline::BSpline, x::Vec)
    otimes_tuple(values.((bspline,), Tuple(x)))
end
@inline otimes_tuple(x::Tuple) = SArray(otimes(map(Vec, x)...))

@inline function Base.values(bspline::BSpline, x::Vec, lattice::Lattice)
    dx⁻¹ = spacing_inv(lattice)
    values(bspline, (x - first(lattice)) * dx⁻¹)
end
@inline function Base.values(bspline::BSpline, pt, lattice::Lattice, ::Symbol)
    x = getx(pt)
    xmin = first(lattice)
    dx⁻¹ = spacing_inv(lattice)
    N, ∇N = values(bspline, (x-xmin)*dx⁻¹, :withgradient)
    (N, ∇N*dx⁻¹)
end

function update_property!(mp::MPValues{<: BSpline}, pt, lattice::Lattice)
    indices = neighbornodes(mp)
    isnearbounds = size(mp.N) != size(indices)
    if isnearbounds
        indices = neighbornodes(mp)
        @inbounds @simd for ip in eachindex(indices)
            i = indices[ip]
            mp.∇N[ip], mp.N[ip] = gradient(x->value(interpolation(mp),x,lattice,i,:steffen), getx(pt), :all)
        end
    else
        map(copyto!, (mp.N, mp.∇N), values(interpolation(mp), pt, lattice, :withgradient))
    end
end
