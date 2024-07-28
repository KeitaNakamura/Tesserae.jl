abstract type AbstractBSpline{order} <: Kernel end

gridspan(::AbstractBSpline{1}) = 2
gridspan(::AbstractBSpline{2}) = 3
gridspan(::AbstractBSpline{3}) = 4

@inline function neighboringnodes(spline::AbstractBSpline, pt, mesh::CartesianMesh{dim}) where {dim}
    x = getx(pt)
    ξ = Tuple(normalize(x, mesh))
    dims = size(mesh)
    isinside(ξ, dims) || return EmptyCartesianIndices(Val(dim))
    offset = _neighboringnodes_offset(spline)
    r = gridspan(spline) - 1
    start = @. unsafe_trunc(Int, floor(ξ - offset)) + 1
    stop = @. start + r
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end
@inline _neighboringnodes_offset(::AbstractBSpline{1}) = 0.0
@inline _neighboringnodes_offset(::AbstractBSpline{2}) = 0.5
@inline _neighboringnodes_offset(::AbstractBSpline{3}) = 1.0

# simple B-spline calculations
function value(::AbstractBSpline{1}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end
function value(::AbstractBSpline{2}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end
function value(::AbstractBSpline{3}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end

@inline fract(x) = x - floor(x)
# Fast calculations for value, gradient and hessian
# `x` must be normalized by `h`
@inline Base.values(spline::AbstractBSpline, x, args...) = only(values(identity, spline, x, args...))
@inline function Base.values(diff, ::AbstractBSpline{1}, x::Real)
    T = typeof(x)
    ξ = fract(x)
    vals = tuple(@. T((1-ξ, ξ)))
    if diff === gradient || diff === hessian || diff === all
        vals = (vals..., @. T((-1, 1)))
    end
    if diff === hessian || diff === all
        vals = (vals..., @. T((0, 0)))
    end
    if diff === all
        vals = (vals..., @. T((0, 0)))
    end
    vals
end
@inline function Base.values(diff, ::AbstractBSpline{2}, x::Real)
    T = typeof(x)
    x′ = fract(x - T(0.5))
    ξ = @. x′ - T((-0.5,0.5,1.5))
    vals = tuple(@. muladd(T((0.5,-1.0,0.5)), ξ^2, muladd(T((-1.5,0.0,1.5)), ξ, T((1.125,0.75,1.125)))))
    if diff === gradient || diff === hessian || diff === all
        vals = (vals..., @. muladd(T((1.0,-2.0,1.0)), ξ, T((-1.5,0.0,1.5))))
    end
    if diff === hessian || diff === all
        vals = (vals..., @. T((1.0,-2.0,1.0)))
    end
    if diff === all
        vals = (vals..., @. T((0.0,0.0,0.0)))
    end
    vals
end
@inline function Base.values(diff, ::AbstractBSpline{3}, x::Real)
    T = typeof(x)
    x′ = fract(x)
    ξ = @. x′ - T((-1,0,1,2))
    ξ² = @. ξ * ξ
    ξ³ = @. ξ² * ξ
    vals = tuple(@. muladd(T((-1/6,0.5,-0.5,1/6)), ξ³, muladd(T((1,-1,-1,1)), ξ², muladd(T((-2,0,0,2)), ξ, T((4/3,2/3,2/3,4/3))))))
    if diff === gradient || diff === hessian || diff === all
        vals = (vals..., @. muladd(T((-0.5,1.5,-1.5,0.5)), ξ², muladd(T((2,-2,-2,2)), ξ, T((-2,0,0,2)))))
    end
    if diff === hessian || diff === all
        vals = (vals..., @. muladd(T((-1.0,3.0,-3.0,1.0)), ξ, T((2,-2,-2,2))))
    end
    if diff === all
        vals = (vals..., @. T((-1.0,3.0,-3.0,1.0)))
    end
    vals
end

@generated function Base.values(diff, spline::AbstractBSpline, x::Vec{dim}) where {dim}
    T_∇∇ws = SymmetricSecondOrderTensor{dim}
    ∇∇ws = Array{Expr}(undef,dim,dim)
    for j in 1:dim, i in 1:dim
        ∇∇ws[i,j] = :(@ntuple $dim α -> α==$j ? (α==$i ? ∇∇x_α : ∇x_α) : ∇ws[$i][α])
    end
    T_∇∇∇ws = Tensor{Tuple{@Symmetry{dim,dim,dim}}}
    ∇∇∇ws = Array{Expr}(undef,dim,dim,dim)
    for k in 1:dim, j in 1:dim, i in 1:dim
        ∇∇∇ws[i,j,k] = :(@ntuple $dim α -> α==$k ? (α==$j ? (α==$i ? ∇∇∇x_α : ∇∇x_α) : ∇x_α) : $∇∇ws[$i,$j][α])
    end
    ∇∇ws = ∇∇ws[Tensorial.indices_unique(T_∇∇ws)]
    ∇∇∇ws = ∇∇∇ws[Tensorial.indices_unique(T_∇∇∇ws)]
    quote
        @_inline_meta
        if diff === identity
            @nexprs $dim d -> (x_d,) = values(identity, spline, x[d])
        elseif diff === gradient
            @nexprs $dim d -> (x_d, ∇x_d) = values(gradient, spline, x[d])
        elseif diff === hessian
            @nexprs $dim d -> (x_d, ∇x_d, ∇∇x_d) = values(hessian, spline, x[d])
        elseif diff === all
            @nexprs $dim d -> (x_d, ∇x_d, ∇∇x_d, ∇∇∇x_d) = values(all, spline, x[d])
        else
            error("wrong diff type, got $diff")
        end

        ws = @ntuple $dim d -> x_d
        wᵢ = tuple_otimes(ws)
        diff === identity && return (wᵢ,)

        ∇ws = @ntuple $dim i -> (@ntuple $dim α -> α==i ? ∇x_α : x_α)
        ∇wᵢ = map(Vec, map(tuple_otimes, ∇ws)...)
        diff === gradient && return (wᵢ,∇wᵢ)

        ∇∇ws = tuple($(∇∇ws...))
        ∇∇wᵢ = map($T_∇∇ws, map(tuple_otimes, ∇∇ws)...)
        diff === hessian && return (wᵢ,∇wᵢ,∇∇wᵢ)

        ∇∇∇ws = tuple($(∇∇∇ws...))
        ∇∇∇wᵢ = map($T_∇∇∇ws, map(tuple_otimes, ∇∇∇ws)...)
        diff === all && return (wᵢ,∇wᵢ,∇∇wᵢ,∇∇∇wᵢ)
    end
end
@inline tuple_otimes(x::Tuple) = SArray(otimes(map(Vec, x)...))

@inline function Base.values(::typeof(identity), spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    h⁻¹ = spacing_inv(mesh)
    (values(spline, (x - get_xmin(mesh)) * h⁻¹),)
end
@inline function Base.values(::typeof(gradient), spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    w, ∇w = values(gradient, spline, (x-xmin)*h⁻¹)
    (w, ∇w*h⁻¹)
end
@inline function Base.values(::typeof(hessian), spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    w, ∇w, ∇∇w = values(hessian, spline, (x-xmin)*h⁻¹)
    (w, ∇w*h⁻¹, ∇∇w*h⁻¹^2)
end
@inline function Base.values(::typeof(all), spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    w, ∇w, ∇∇w, ∇∇∇w = values(all, spline, (x-xmin)*h⁻¹)
    (w, ∇w*h⁻¹, ∇∇w*h⁻¹^2, ∇∇∇w*h⁻¹^3)
end

function update_property!(mp::MPValue, it::AbstractBSpline, pt, mesh::CartesianMesh)
    indices = neighboringnodes(mp)
    isnearbounds = size(mp.w) != size(indices)
    if isnearbounds
        @inbounds for ip in eachindex(indices)
            i = indices[ip]
            set_kernel_values!(mp, ip, value(difftype(mp), it, getx(pt), mesh, i))
        end
    else
        set_kernel_values!(mp, values(difftype(mp), it, getx(pt), mesh))
    end
end

struct BSpline{order} <: AbstractBSpline{order}
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
"""
const QuadraticBSpline = BSpline{2}

"""
    BSpline{3}()
    CubicBSpline()

Cubic B-spline kernel.
"""
const CubicBSpline = BSpline{3}

@generated function value(spline::BSpline, ξ::Vec{dim}) where {dim}
    quote
        @_inline_meta
        prod(@ntuple $dim i -> value(spline, ξ[i]))
    end
end
@inline function value(spline::BSpline, pt, mesh::CartesianMesh, i)
    @_propagate_inbounds_meta
    xₚ = getx(pt)
    xᵢ = mesh[i]
    h⁻¹ = spacing_inv(mesh)
    ξ = (xₚ - xᵢ) * h⁻¹
    value(spline, ξ)
end

struct SteffenBSpline{order} <: AbstractBSpline{order}
    SteffenBSpline{order}() where {order} = new{order::Int}()
end

"""
    SteffenBSpline{1}()
    SteffenLinearBSpline()

Linear B-spline kernel.
"""
const SteffenLinearBSpline = SteffenBSpline{1}

"""
    SteffenBSpline{2}()
    SteffenQuadraticBSpline()

Quadratic B-spline kernel proposed by [^Steffen].

[^Steffen]: [Steffen, M., Kirby, R. M., & Berzins, M. (2008). Analysis and reduction of quadrature errors in the material point method (MPM). *International journal for numerical methods in engineering*, 76(6), 922-948.](https://doi.org/10.1002/nme.2360)
"""
const SteffenQuadraticBSpline = SteffenBSpline{2}

"""
    SteffenBSpline{3}()
    SteffenCubicBSpline()

Cubic B-spline kernel proposed by [^Steffen].

[^Steffen]: [Steffen, M., Kirby, R. M., & Berzins, M. (2008). Analysis and reduction of quadrature errors in the material point method (MPM). *International journal for numerical methods in engineering*, 76(6), 922-948.](https://doi.org/10.1002/nme.2360)
"""
const SteffenCubicBSpline = SteffenBSpline{3}

# Steffen, M., Kirby, R. M., & Berzins, M. (2008).
# Analysis and reduction of quadrature errors in the material point method (MPM).
# International journal for numerical methods in engineering, 76(6), 922-948.
function value(spline::SteffenBSpline{1}, ξ::Real, pos::Int)::typeof(ξ)
    value(spline, ξ)
end
function value(spline::SteffenBSpline{2}, ξ::Real, pos::Int)::typeof(ξ)
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
function value(spline::SteffenBSpline{3}, ξ::Real, pos::Int)::typeof(ξ)
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
@generated function value(spline::SteffenBSpline, ξ::Vec{dim}, pos::Vec{dim}) where {dim}
    quote
        @_inline_meta
        prod(@ntuple $dim i -> value(spline, ξ[i], pos[i]))
    end
end
@inline function value(spline::SteffenBSpline, pt, mesh::CartesianMesh, i::CartesianIndex)
    @_propagate_inbounds_meta
    xₚ = getx(pt)
    xᵢ = mesh[i]
    h⁻¹ = spacing_inv(mesh)
    ξ = (xₚ - xᵢ) * h⁻¹
    value(spline, ξ, node_position(mesh, i))
end

@inline function node_position(ax::AbstractVector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
node_position(mesh::CartesianMesh, index::CartesianIndex) = Vec(map(node_position, get_axes(mesh), Tuple(index)))
