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

@pure function maxnum_nodes(bspline::BSpline, ::Val{dim})::Int where {dim}
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
@inline value(bspline::BSpline, ξ::Vec) = prod(value.(bspline, ξ))
function value(bspline::BSpline, grid::Grid, I::CartesianIndex, xp::Vec)
    @_propagate_inbounds_meta
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    value(bspline, ξ)
end
@inline value(bspline::BSpline, grid::Grid, I::CartesianIndex, pt) = value(bspline, grid, I, pt.x)

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
@inline value(bspline::BSpline, ξ::Vec, pos::Tuple{Vararg{Int}}) = prod(value.(bspline, ξ, pos))
function value(bspline::BSpline, grid::Grid, I::CartesianIndex, xp::Vec, ::Symbol) # last argument is pseudo argument `:steffen`
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    value(bspline, ξ, node_position(grid, I))
end

@inline function node_position(ax::Vector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
node_position(grid::Grid, index::CartesianIndex) = map(node_position, gridaxes(grid), Tuple(index))


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
@generated function simd_otimes(x::SVec{m}, y::SVec{n}) where {m, n}
    exps = [:($(Symbol(:z_,j))[$i]) for j in 1:n for i in 1:m]
    quote
        @_inline_meta
        @nexprs $n j -> z_j = x * y[j]
        SVec(tuple($(exps...)))
    end
end
simd_otimes(x::SVec, y::SVec, others::SVec...) = simd_otimes(simd_otimes(x, y), others...)
simd_otimes(x::SVec) = x
@generated function values_gradients(bspline::BSpline, grid::Grid{dim}, xp::Vec{dim}) where {dim}
    quote
        @_inline_meta
        dx⁻¹ = gridsteps_inv(grid)
        x = (xp - first(grid)) .* dx⁻¹
        vals_grads = @ntuple $dim d -> values_gradients(bspline, x[d])
        vals  = getindex.(vals_grads, 1)
        grads = getindex.(vals_grads, 2)
        Tuple(simd_otimes(vals...)), Vec.((@ntuple $dim i -> begin
                                               Tuple(simd_otimes((@ntuple $dim d -> d==i ? grads[d]*dx⁻¹[d] : vals[d])...))
                                           end)...)
    end
end
@inline values_gradients(bspline::BSpline, grid::Grid, pt) = values_gradients(bspline, grid, pt.x)


struct BSplineValue{dim, T, order} <: MPValue{dim, T, BSpline{order}}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function MPValue{dim, T}(::BSpline{order}) where {dim, T, order}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    BSplineValue{dim, T, order}(N, ∇N)
end

@inline function update!(mp::BSplineValue, ::NearBoundary{false}, grid::Grid, ::AllTrue, nodeinds::CartesianIndices, xp::Vec)
    n = length(nodeinds)
    resize!(mp.N, n)
    resize!(mp.∇N, n)
    wᵢ, ∇wᵢ = values_gradients(get_kernel(mp), grid, xp)
    mp.N .= wᵢ
    mp.∇N .= ∇wᵢ
    mp
end

function update!(mp::BSplineValue, ::NearBoundary{true}, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::CartesianIndices, xp::Vec)
    n = length(nodeinds)
    resize!(mp.N, n)
    resize!(mp.∇N, n)
    F = get_kernel(mp)
    @inbounds for (j, i) in enumerate(nodeinds)
        mp.∇N[j], mp.N[j] = gradient(x->value(F,grid,i,x,:steffen), xp, :all) .* sppat[i]
    end
    mp
end
