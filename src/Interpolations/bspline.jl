struct BSpline{order} <: Kernel
    BSpline{order}() where {order} = new{order::Int}()
end

const LinearBSpline    = BSpline{1}
const QuadraticBSpline = BSpline{2}
const CubicBSpline     = BSpline{3}

@inline neighbornodes(::BSpline{1}, lattice::Lattice, pt) = neighbornodes(lattice, getx(pt), 1.0)
@inline neighbornodes(::BSpline{2}, lattice::Lattice, pt) = neighbornodes(lattice, getx(pt), 1.5)
@inline neighbornodes(::BSpline{3}, lattice::Lattice, pt) = neighbornodes(lattice, getx(pt), 2.0)
@inline neighbornodes(::BSpline{4}, lattice::Lattice, pt) = neighbornodes(lattice, getx(pt), 2.5)

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
function value(::BSpline{4}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (48ξ^4 - 120ξ^2 + 115) / 192 :
    ξ < 1.5 ? -(16ξ^4 - 80ξ^3 + 120ξ^2 - 20ξ - 55) / 96 :
    ξ < 2.5 ? (5 - 2ξ)^4 / 384 : zero(ξ)
end
@generated function value(bspline::BSpline, ξ::Vec{dim}) where {dim}
    quote
        @_inline_meta
        prod(@ntuple $dim i -> value(bspline, ξ[i]))
    end
end
@inline function value(bspline::BSpline, lattice::Lattice, I::CartesianIndex, xp::Vec)
    @_propagate_inbounds_meta
    xi = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xp - xi) * dx⁻¹
    value(bspline, ξ)
end
@inline function value(bspline::BSpline, lattice::Lattice, I::CartesianIndex, pt)
    @_propagate_inbounds_meta
    value(bspline, lattice, I, getx(pt))
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
@inline function value(bspline::BSpline, lattice::Lattice, I::CartesianIndex, xp::Vec, ::Symbol) # last argument is p
    @_propagate_inbounds_meta
    xi = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xp - xi) * dx⁻¹
    value(bspline, ξ, node_position(lattice, I))
end

@inline function node_position(ax::Vector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
node_position(lattice::Lattice, index::CartesianIndex) = map(node_position, get_axes(lattice), Tuple(index))


fract(x) = x - floor(x)
# Fast calculations for values and gradients
# `x` must be normalized by `dx`
@inline function values_gradients(::BSpline{1}, x::SIMDTypes)
    T = typeof(x)
    V = SVec{2, T}
    ξ = fract(x)
    V((1-ξ, ξ)), V((-1, 1))
end
@inline function values_gradients(::BSpline{2}, x::SIMDTypes)
    T = typeof(x)
    V = SVec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ - V((-0.5,0.5,1.5))
    vals = V((0.5,-1.0,0.5))*ξ^2 + V((-1.5,0.0,1.5))*ξ + V((1.125,0.75,1.125))
    grads = V((1.0,-2.0,1.0))*ξ + V((-1.5,0.0,1.5))
    vals, grads
end
@inline function values_gradients(::BSpline{3}, x::SIMDTypes)
    T = typeof(x)
    V = SVec{4, T}
    x′ = fract(x)
    ξ = x′ - V((-1,0,1,2))
    ξ² = ξ * ξ
    ξ³ = ξ² * ξ
    vals = V((-1/6,0.5,-0.5,1/6))*ξ³ + V((1,-1,-1,1))*ξ² + V((-2,0,0,2))*ξ + V((4/3,2/3,2/3,4/3))
    grads = V((-0.5,1.5,-1.5,0.5))*ξ² + V((2,-2,-2,2))*ξ + V((-2,0,0,2))
    vals, grads
end

@inline values_gradients!(N, ∇N, bspline::BSpline, lattice::Lattice, pt) = values_gradients!(N, ∇N, bspline, lattice, getx(pt))

@generated function values_gradients!(N, ∇N, bspline::BSpline, lattice::Lattice{dim}, xp::Vec{dim}) where {dim}
    quote
        @_inline_meta
        dx⁻¹ = spacing_inv(lattice)
        x = (xp - first(lattice)) * dx⁻¹
        @nexprs $dim d -> (V_d, ∇V_d) = values_gradients(bspline, x[d])
        V_tuple = @ntuple $dim d -> SVector(V_d)
        ∇V_tuple = @ntuple $dim d -> SVector(∇V_d*dx⁻¹)
        _values_gradients!(N, ∇N, V_tuple, ∇V_tuple)
    end
end

# 1D
@inline function _values_gradients!(N, ∇N, (Vx,)::NTuple{1}, (∇Vx,)::NTuple{1})
    N .= Vx
    ∇N .= ∇Vx
end

# 2D
@inline function _values_gradients!(N, ∇N, (Vx, Vy)::NTuple{2}, (∇Vx, ∇Vy)::NTuple{2})
    n = length(Vx)
    @turbo for j in 1:n
        offset = n*(j-1)
        for i in 1:n
            N[offset+i] = Vx[i] * Vy[j]
            ∇N[1, offset+i] = ∇Vx[i] *  Vy[j]
            ∇N[2, offset+i] =  Vx[i] * ∇Vy[j]
        end
    end
end

# 3D
@inline function _values_gradients!(N, ∇N, (Vx, Vy, Vz)::NTuple{3}, (∇Vx, ∇Vy, ∇Vz)::NTuple{3})
    n = length(Vx)
    @turbo for k in 1:n
        offset_j = n*n*(k-1)
        for j in 1:n
            offset_i = offset_j + n*(j-1)
            for i in 1:n
                N[offset_i+i] = Vx[i] * Vy[j] * Vz[k]
                ∇N[1, offset_i+i] = ∇Vx[i] *  Vy[j] *  Vz[k]
                ∇N[2, offset_i+i] =  Vx[i] * ∇Vy[j] *  Vz[k]
                ∇N[3, offset_i+i] =  Vx[i] *  Vy[j] * ∇Vz[k]
            end
        end
    end
end

struct BSplineValue{dim, T, order} <: MPValue{dim, T}
    itp::BSpline{order}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function MPValue{dim, T}(itp::BSpline{order}) where {dim, T, order}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    BSplineValue(itp, N, ∇N)
end

num_nodes(mp::BSplineValue) = length(mp.N)
@inline shape_value(mp::BSplineValue, j::Int) = (@_propagate_inbounds_meta; mp.N[j])
@inline shape_gradient(mp::BSplineValue, j::Int) = (@_propagate_inbounds_meta; mp.∇N[j])

@inline function update_mpvalue!(mp::BSplineValue{<: Any, T}, lattice::Lattice, pt) where {T}
    indices, isfullyinside = neighbornodes(mp.itp, lattice, pt)

    n = length(indices)
    resize!(mp.N, n)
    resize!(mp.∇N, n)

    if isfullyinside
        N = mp.N
        ∇N = reinterpret(reshape, T, mp.∇N)
        values_gradients!(N, ∇N, mp.itp, lattice, pt)
    else
        update_mpvalue_nearbounds!(mp, lattice, indices, pt)
    end

    indices
end

function update_mpvalue_nearbounds!(mp::BSplineValue, lattice::Lattice, indices, pt)
    @inbounds for (j, i) in pairs(IndexLinear(), indices)
        mp.∇N[j], mp.N[j] = gradient(x->value(mp.itp,lattice,i,x,:steffen), getx(pt), :all)
    end
end
