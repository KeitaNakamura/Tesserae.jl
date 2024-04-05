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

@inline function neighboringnodes(bspline::BSpline, pt, mesh::CartesianMesh{dim}) where {dim}
    x = getx(pt)
    ξ = Tuple(normalize(x, mesh))
    dims = size(mesh)
    isinside(ξ, dims) || return ZeroCartesianIndices(Val(dim))
    offset = _neighboringnodes_offset(bspline)
    h = gridspan(bspline) - 1
    start = @. unsafe_trunc(Int, floor(ξ - offset)) + 1
    stop = @. start + h
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end
@inline _neighboringnodes_offset(::BSpline{1}) = 0.0
@inline _neighboringnodes_offset(::BSpline{2}) = 0.5
@inline _neighboringnodes_offset(::BSpline{3}) = 1.0

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
@inline function value(bspline::BSpline, xₚ::Vec, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    xᵢ = mesh[I]
    dx⁻¹ = spacing_inv(mesh)
    ξ = (xₚ - xᵢ) * dx⁻¹
    value(bspline, ξ)
end
@inline function value(bspline::BSpline, pt, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    value(bspline, getx(pt), mesh, I)
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
@inline function value(bspline::BSpline, xₚ::Vec, mesh::CartesianMesh, I::CartesianIndex, ::Symbol) # last argument is pseudo argument `:steffen`
    @_propagate_inbounds_meta
    xᵢ = mesh[I]
    dx⁻¹ = spacing_inv(mesh)
    ξ = (xₚ - xᵢ) * dx⁻¹
    value(bspline, ξ, node_position(mesh, I))
end

@inline function node_position(ax::AbstractVector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
node_position(mesh::CartesianMesh, index::CartesianIndex) = map(node_position, get_axes(mesh), Tuple(index))


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

@inline function Base.values(bspline::BSpline, x::Vec, mesh::CartesianMesh)
    dx⁻¹ = spacing_inv(mesh)
    values(bspline, (x - first(mesh)) * dx⁻¹)
end
@inline function Base.values(bspline::BSpline, pt, mesh::CartesianMesh, ::Symbol)
    x = getx(pt)
    xmin = first(mesh)
    dx⁻¹ = spacing_inv(mesh)
    N, ∇N = values(bspline, (x-xmin)*dx⁻¹, :withgradient)
    (N, ∇N*dx⁻¹)
end

function update_property!(mp::MPValues{<: BSpline}, pt, mesh::CartesianMesh)
    indices = neighboringnodes(mp)
    isnearbounds = size(mp.N) != size(indices)
    if isnearbounds
        @inbounds @simd for ip in eachindex(indices)
            i = indices[ip]
            mp.∇N[ip], mp.N[ip] = gradient(x->value(interpolation(mp),x,mesh,i,:steffen), getx(pt), :all)
        end
    else
        map(copyto!, (mp.N, mp.∇N), values(interpolation(mp), pt, mesh, :withgradient))
    end
end
