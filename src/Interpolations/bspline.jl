abstract type AbstractBSpline{D <: Degree} <: Kernel end

gridspan(::AbstractBSpline{Degree{1}}) = 2
gridspan(::AbstractBSpline{Degree{2}}) = 3
gridspan(::AbstractBSpline{Degree{3}}) = 4

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
@inline _neighboringnodes_offset(::AbstractBSpline{Degree{1}}) = 0.0
@inline _neighboringnodes_offset(::AbstractBSpline{Degree{2}}) = 0.5
@inline _neighboringnodes_offset(::AbstractBSpline{Degree{3}}) = 1.0

# simple B-spline calculations
function value(::AbstractBSpline{Linear}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end
function value(::AbstractBSpline{Quadratic}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end
function value(::AbstractBSpline{Cubic}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end

@inline fract(x) = x - floor(x)
# Fast calculations for value, gradient and hessian
# `x` must be normalized by `h`
@inline Base.values(spline::AbstractBSpline, x, args...) = only(values(Order(0), spline, x, args...))
@inline function Base.values(order, ::AbstractBSpline{Linear}, x::Real)
    T = typeof(x)
    ξ = fract(x)
    vals = tuple(@. T((1-ξ, ξ)))
    if order isa Union{Order{1}, Order{2}, Order{3}}
        vals = (vals..., @. T((-1, 1)))
    end
    if order isa Union{Order{2}, Order{3}}
        vals = (vals..., @. T((0, 0)))
    end
    if order isa Order{3}
        vals = (vals..., @. T((0, 0)))
    end
    vals
end
@inline function Base.values(order, ::AbstractBSpline{Quadratic}, x::Real)
    T = typeof(x)
    x′ = fract(x - T(0.5))
    ξ = @. x′ - T((-0.5,0.5,1.5))
    vals = tuple(@. muladd(T((0.5,-1.0,0.5)), ξ^2, muladd(T((-1.5,0.0,1.5)), ξ, T((1.125,0.75,1.125)))))
    if order isa Union{Order{1}, Order{2}, Order{3}}
        vals = (vals..., @. muladd(T((1.0,-2.0,1.0)), ξ, T((-1.5,0.0,1.5))))
    end
    if order isa Union{Order{2}, Order{3}}
        vals = (vals..., @. T((1.0,-2.0,1.0)))
    end
    if order isa Order{3}
        vals = (vals..., @. T((0.0,0.0,0.0)))
    end
    vals
end
@inline function Base.values(order, ::AbstractBSpline{Cubic}, x::Real)
    T = typeof(x)
    x′ = fract(x)
    ξ = @. x′ - T((-1,0,1,2))
    ξ² = @. ξ * ξ
    ξ³ = @. ξ² * ξ
    vals = tuple(@. muladd(T((-1/6,0.5,-0.5,1/6)), ξ³, muladd(T((1,-1,-1,1)), ξ², muladd(T((-2,0,0,2)), ξ, T((4/3,2/3,2/3,4/3))))))
    if order isa Union{Order{1}, Order{2}, Order{3}}
        vals = (vals..., @. muladd(T((-0.5,1.5,-1.5,0.5)), ξ², muladd(T((2,-2,-2,2)), ξ, T((-2,0,0,2)))))
    end
    if order isa Union{Order{2}, Order{3}}
        vals = (vals..., @. muladd(T((-1.0,3.0,-3.0,1.0)), ξ, T((2,-2,-2,2))))
    end
    if order isa Order{3}
        vals = (vals..., @. T((-1.0,3.0,-3.0,1.0)))
    end
    vals
end

@generated function Base.values(order, spline::AbstractBSpline, x::Vec{dim}) where {dim}
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
        if order isa Order{0}
            @nexprs $dim d -> (x_d,) = values(order, spline, x[d])
        elseif order isa Order{1}
            @nexprs $dim d -> (x_d, ∇x_d) = values(order, spline, x[d])
        elseif order isa Order{2}
            @nexprs $dim d -> (x_d, ∇x_d, ∇∇x_d) = values(order, spline, x[d])
        elseif order isa Order{3}
            @nexprs $dim d -> (x_d, ∇x_d, ∇∇x_d, ∇∇∇x_d) = values(order, spline, x[d])
        else
            error("wrong order, got $order")
        end

        ws = @ntuple $dim d -> x_d
        wᵢ = tuple_otimes(ws)
        order isa Order{0} && return (wᵢ,)

        ∇ws = @ntuple $dim i -> (@ntuple $dim α -> α==i ? ∇x_α : x_α)
        ∇wᵢ = map(Vec, map(tuple_otimes, ∇ws)...)
        order isa Order{1} && return (wᵢ,∇wᵢ)

        ∇∇ws = tuple($(∇∇ws...))
        ∇∇wᵢ = map($T_∇∇ws, map(tuple_otimes, ∇∇ws)...)
        order isa Order{2} && return (wᵢ,∇wᵢ,∇∇wᵢ)

        ∇∇∇ws = tuple($(∇∇∇ws...))
        ∇∇∇wᵢ = map($T_∇∇∇ws, map(tuple_otimes, ∇∇∇ws)...)
        order isa Order{3} && return (wᵢ,∇wᵢ,∇∇wᵢ,∇∇∇wᵢ)
    end
end
@inline tuple_otimes(x::Tuple) = SArray(otimes(map(Vec, x)...))

@inline function Base.values(::Order{0}, spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    h⁻¹ = spacing_inv(mesh)
    (values(spline, (x - get_xmin(mesh)) * h⁻¹),)
end
@inline function Base.values(order::Order{1}, spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    w, ∇w = values(order, spline, (x-xmin)*h⁻¹)
    (w, ∇w*h⁻¹)
end
@inline function Base.values(order::Order{2}, spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    w, ∇w, ∇∇w = values(order, spline, (x-xmin)*h⁻¹)
    (w, ∇w*h⁻¹, ∇∇w*h⁻¹^2)
end
@inline function Base.values(order::Order{3}, spline::AbstractBSpline, x::Vec, mesh::CartesianMesh)
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    w, ∇w, ∇∇w, ∇∇∇w = values(order, spline, (x-xmin)*h⁻¹)
    (w, ∇w*h⁻¹, ∇∇w*h⁻¹^2, ∇∇∇w*h⁻¹^3)
end

function update_property!(mp::MPValue, it::AbstractBSpline, pt, mesh::CartesianMesh)
    indices = neighboringnodes(mp)
    isnearbounds = size(mp.w) != size(indices)
    if isnearbounds
        @inbounds for ip in eachindex(indices)
            i = indices[ip]
            set_kernel_values!(mp, ip, value(derivative_order(mp), it, getx(pt), mesh, i))
        end
    else
        set_kernel_values!(mp, values(derivative_order(mp), it, getx(pt), mesh))
    end
end

"""
    BSpline(degree)

B-spline kernel.
"""
struct BSpline{D} <: AbstractBSpline{D}
    degree::D
end
Base.show(io::IO, spline::BSpline) = print(io, BSpline, "(", spline.degree, ")")

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

"""
    BSpline(degree)

B-spline kernel with boundary treatments by [^Steffen].

[^Steffen]: [Steffen, M., Kirby, R. M., & Berzins, M. (2008). Analysis and reduction of quadrature errors in the material point method (MPM). *International journal for numerical methods in engineering*, 76(6), 922-948.](https://doi.org/10.1002/nme.2360)
"""
struct SteffenBSpline{D} <: AbstractBSpline{D}
    degree::D
end
Base.show(io::IO, spline::SteffenBSpline) = print(io, SteffenBSpline, "(", spline.degree, ")")

# Steffen, M., Kirby, R. M., & Berzins, M. (2008).
# Analysis and reduction of quadrature errors in the material point method (MPM).
# International journal for numerical methods in engineering, 76(6), 922-948.
function value(spline::SteffenBSpline{Linear}, ξ::Real, pos::Int)::typeof(ξ)
    value(spline, ξ)
end
function value(spline::SteffenBSpline{Quadratic}, ξ::Real, pos::Int)::typeof(ξ)
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
function value(spline::SteffenBSpline{Cubic}, ξ::Real, pos::Int)::typeof(ξ)
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
