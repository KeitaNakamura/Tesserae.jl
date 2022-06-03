"""
    BSpline{order}()
    LinearBSpline()
    QuadraticBSpline()
    CubicBSpline()

Create B-spline kernel.

# Examples
```jldoctest
julia> f = LinearBSpline()
LinearBSpline()

julia> Marble.value(f, Vec(0.5, 0.5))
0.25
```
"""
struct BSpline{order} <: Kernel
    function BSpline{order}() where {order}
        new{order::Int}()
    end
end

const LinearBSpline    = BSpline{1}
const QuadraticBSpline = BSpline{2}
const CubicBSpline     = BSpline{3}

get_supportlength(::BSpline{1}) = 1.0
get_supportlength(::BSpline{2}) = 1.5
get_supportlength(::BSpline{3}) = 2.0
get_supportlength(::BSpline{4}) = 2.5

@pure function num_nodes(bspline::BSpline, ::Val{dim})::Int where {dim}
    (2*get_supportlength(bspline))^dim
end

@inline function neighbornodes(bsp::BSpline, grid::Grid, x::Vec)
    neighbornodes(grid, x, get_supportlength(bsp))
end
@inline neighbornodes(bsp::BSpline, grid::Grid, pt) = neighbornodes(bsp, grid, pt.x)

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
@inline value(bspline::BSpline, ξ::Vec) = prod(map_tuple(value, bspline, Tuple(ξ)))
# used in `WLS`
function value(bspline::BSpline, grid::Grid, I::Index, xp::Vec)
    @_inline_propagate_inbounds_meta
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    value(bspline, ξ)
end
@inline value(bspline::BSpline, grid::Grid, I::Index, pt) = value(bspline::BSpline, grid::Grid, I::Index, pt.x)
# used in `KernelCorrection`
function value_gradient(bspline::BSpline, grid::Grid, I::Index, xp::Vec)
    @_inline_propagate_inbounds_meta
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    ∇w, w = gradient(ξ -> value(bspline, ξ), ξ, :all)
    w, ∇w.*dx⁻¹
end
@inline value_gradient(bspline::BSpline, grid::Grid, I::Index, pt) = value_gradient(bspline::BSpline, grid::Grid, I::Index, pt.x)

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
@inline value(bspline::BSpline, ξ::Vec, pos::Tuple{Vararg{Int}}) = prod(map_tuple(value, bspline, Tuple(ξ), pos))
function value_gradient(bspline::BSpline, grid::Grid, I::Index, xp::Vec, ::Symbol) # last argument is pseudo argument `:steffen`
    @_inline_propagate_inbounds_meta
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    ∇w, w = gradient(ξ -> value(bspline, ξ, node_position(grid, I)), ξ, :all)
    w, ∇w.*dx⁻¹
end


fract(x) = x - floor(x)
# Fast calculations for values
# used in `WLS`
# `x` must be normalized by `dx`
function Base.values(::BSpline{1}, x::T) where {T <: Real}
    ξ = fract(x)
    Vec{2, T}(1-ξ, ξ)
end
function Base.values(::BSpline{2}, x::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    @. $V(0.5,-1.0,0.5)*ξ^2 + $V(-1.5,0.0,1.5)*ξ + $V(1.125, 0.75, 1.125)
end
function Base.values(::BSpline{3}, x::T) where {T <: Real}
    V = Vec{4, T}
    x′ = fract(x)
    ξ = x′ .- V(-1, 0, 1, 2)
    ξ² = ξ .* ξ
    ξ³ = ξ² .* ξ
    @. $V(-1/6,0.5,-0.5,1/6)*ξ³ + $V(1,-1,-1,1)*ξ² + $V(-2,0,0,2)*ξ + $V(4/3,2/3,2/3,4/3)
end
@inline Base.values(bspline::BSpline, x::Vec) = Tuple(otimes(map_tuple(values, bspline, Tuple(x))...))
function Base.values(bspline::BSpline, grid::Grid, xp::Vec)
    dx⁻¹ = gridsteps_inv(grid)
    values(bspline, xp .* dx⁻¹)
end
@inline Base.values(bspline::BSpline, grid::Grid, pt) = values(bspline, grid, pt.x)

# Fast calculations for values and gradients
# used in `KernelCorrection`
# `x` must be normalized by `dx`
function values_gradients(::BSpline{1}, x::T) where {T <: Real}
    V = Vec{2, T}
    ξ = fract(x)
    V(1-ξ, ξ), V(-1, 1)
end
function values_gradients(::BSpline{2}, x::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    vals = @. $V(0.5,-1.0,0.5)*ξ^2 + $V(-1.5,0.0,1.5)*ξ + $V(1.125,0.75,1.125)
    grads = @. $V(1.0,-2.0,1.0)*ξ + $V(-1.5,0.0,1.5)
    vals, grads
end
function values_gradients(::BSpline{3}, x::T) where {T <: Real}
    V = Vec{4, T}
    x′ = fract(x)
    ξ = x′ .- V(-1, 0, 1, 2)
    ξ² = ξ .* ξ
    ξ³ = ξ² .* ξ
    vals = @. $V(-1/6,0.5,-0.5,1/6)*ξ³ + $V(1,-1,-1,1)*ξ² + $V(-2,0,0,2)*ξ + $V(4/3,2/3,2/3,4/3)
    grads = @. $V(-0.5,1.5,-1.5,0.5)*ξ² + $V(2,-2,-2,2)*ξ + $V(-2,0,0,2)
    vals, grads
end
@generated function values_gradients(bspline::BSpline, x::Vec{dim}) where {dim}
    exps = map(1:dim) do i
        x = [d == i ? :(grads[$d]) : :(vals[$d]) for d in 1:dim]
        :(Tuple(otimes($(x...))))
    end
    quote
        @_inline_meta
        vals_grads = map_tuple(values_gradients, bspline, Tuple(x))
        vals  = map_tuple(getindex, vals_grads, 1)
        grads = map_tuple(getindex, vals_grads, 2)
        Tuple(otimes(vals...)), map_tuple(Vec, $(exps...))
    end
end
function values_gradients(bspline::BSpline, grid::Grid, xp::Vec)
    dx⁻¹ = gridsteps_inv(grid)
    wᵢ, ∇wᵢ = values_gradients(bspline, xp .* dx⁻¹)
    wᵢ, broadcast(.*, ∇wᵢ, Ref(dx⁻¹))
end
@inline values_gradients(bspline::BSpline, grid::Grid, pt) = values_gradients(bspline, grid, pt.x)


struct BSplineValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    xp::Vec{dim, T}
end

mutable struct BSplineValues{order, dim, T, L} <: MPValues{dim, T, BSplineValue{dim, T}}
    F::BSpline{order}
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    gridindices::MVector{L, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

# constructors
function BSplineValues{order, dim, T, L}() where {order, dim, T, L}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    gridindices = MVector{L, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    BSplineValues(BSpline{order}(), N, ∇N, gridindices, xp, 0)
end
function MPValues{dim, T}(F::BSpline{order}) where {order, dim, T}
    L = num_nodes(F, Val(dim))
    BSplineValues{order, dim, T, L}()
end

get_kernel(x::BSplineValues) = x.F

@inline function node_position(ax::Vector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
node_position(grid::Grid, index::Index) = map(node_position, gridaxes(grid), Tuple(index.I))

function update!(mpvalues::BSplineValues, grid::Grid, xp::Vec, spat::AbstractArray{Bool})
    # reset
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)

    F = get_kernel(mpvalues)

    # update
    mpvalues.xp = xp
    update_active_gridindices!(mpvalues, neighbornodes(F, grid, xp), spat)
    @inbounds @simd for i in 1:length(mpvalues)
        I = gridindices(mpvalues, i)
        mpvalues.N[i], mpvalues.∇N[i] = value_gradient(F, grid, I, xp, :steffen)
    end
    mpvalues
end

@inline function Base.getindex(mpvalues::BSplineValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.xp)
end
