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
@inline function values_gradients(::BSpline{1}, x::Real)
    T = typeof(x)
    V = NTuple{2, T}
    ξ = fract(x)
    V((1-ξ, ξ)), V((-1, 1))
end
@inline function values_gradients(::BSpline{2}, x::Real)
    T = typeof(x)
    V = NTuple{3, T}
    x′ = fract(x - T(0.5))
    ξ = @. x′ - $V((-0.5,0.5,1.5))
    vals = @. $V((0.5,-1.0,0.5))*ξ^2 + $V((-1.5,0.0,1.5))*ξ + $V((1.125,0.75,1.125))
    grads = @. $V((1.0,-2.0,1.0))*ξ + $V((-1.5,0.0,1.5))
    vals, grads
end
@inline function values_gradients(::BSpline{3}, x::Real)
    T = typeof(x)
    V = NTuple{4, T}
    x′ = fract(x)
    ξ = @. x′ - $V((-1,0,1,2))
    ξ² = @. ξ * ξ
    ξ³ = @. ξ² * ξ
    vals = @. $V((-1/6,0.5,-0.5,1/6))*ξ³ + $V((1,-1,-1,1))*ξ² + $V((-2,0,0,2))*ξ + $V((4/3,2/3,2/3,4/3))
    grads = @. $V((-0.5,1.5,-1.5,0.5))*ξ² + $V((2,-2,-2,2))*ξ + $V((-2,0,0,2))
    vals, grads
end

@inline values_gradients!(N, ∇N, bspline::BSpline, pt, lattice::Lattice) = values_gradients!(N, ∇N, bspline, getx(pt), lattice)

@generated function values_gradients!(N, ∇N, bspline::BSpline, xₚ::Vec{dim}, lattice::Lattice{dim}) where {dim}
    quote
        @_inline_meta
        dx⁻¹ = spacing_inv(lattice)
        x = (xₚ - first(lattice)) * dx⁻¹
        @nexprs $dim d -> (V_d, ∇V_d) = values_gradients(bspline, x[d])
        V_tuple = @ntuple $dim d -> V_d
        ∇V_tuple = @ntuple $dim d -> ∇V_d.*dx⁻¹
        _values_gradients!(N, reinterpret(reshape, eltype(eltype(∇N)), ∇N), V_tuple, ∇V_tuple)
    end
end

# 1D
@inline function _values_gradients!(N, ∇N, (Vx,)::NTuple{1}, (∇Vx,)::NTuple{1})
    N .= Tuple(Vx)
    ∇N .= Tuple(∇Vx)
end

# 2D
@inline function _values_gradients!(N, ∇N, (Vx, Vy)::NTuple{2}, (∇Vx, ∇Vy)::NTuple{2})
    n = length(Vx)
    @inbounds for j in 1:n
        @simd for i in 1:n
            N[i,j] = Vx[i] * Vy[j]
            ∇N[1,i,j] = ∇Vx[i] *  Vy[j]
            ∇N[2,i,j] =  Vx[i] * ∇Vy[j]
        end
    end
end

# 3D
@inline function _values_gradients!(N, ∇N, (Vx, Vy, Vz)::NTuple{3}, (∇Vx, ∇Vy, ∇Vz)::NTuple{3})
    n = length(Vx)
    @inbounds for k in 1:n
        for j in 1:n
            @simd for i in 1:n
                N[i,j,k] = Vx[i] * Vy[j] * Vz[k]
                ∇N[1,i,j,k] = ∇Vx[i] *  Vy[j] *  Vz[k]
                ∇N[2,i,j,k] =  Vx[i] * ∇Vy[j] *  Vz[k]
                ∇N[3,i,j,k] =  Vx[i] *  Vy[j] * ∇Vz[k]
            end
        end
    end
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
        values_gradients!(mp.N, mp.∇N, interpolation(mp), pt, lattice)
    end
end
