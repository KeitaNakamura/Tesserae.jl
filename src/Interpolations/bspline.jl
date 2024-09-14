abstract type AbstractBSpline{D <: Degree} <: Kernel end

gridspan(::AbstractBSpline{Degree{1}}) = 2
gridspan(::AbstractBSpline{Degree{2}}) = 3
gridspan(::AbstractBSpline{Degree{3}}) = 4
gridspan(::AbstractBSpline{Degree{4}}) = 5

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
@inline _neighboringnodes_offset(::AbstractBSpline{Degree{4}}) = 1.5

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
function value(::AbstractBSpline{Quartic}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (48ξ^4 - 120ξ^2 + 115) / 192              :
    ξ < 1.5 ? -(16ξ^4 - 80ξ^3 + 120ξ^2 - 20ξ - 55) / 96 :
    ξ < 2.5 ? (5 - 2ξ)^4 / 384                          : zero(ξ)
end

@inline fract(x) = x - floor(x)
# Fast calculations for value, gradient and hessian
# `x` must be normalized by `h`
@inline Base.values(spline::AbstractBSpline, x, args...) = only(values(Order(0), spline, x, args...))

# linear
@inline _values(::Order{0}, ::AbstractBSpline{Linear}, (ξ,)::Tuple{V}) where {V} = @. muladd($V((-1,1)), ξ, $V((1,1)))
@inline _values(::Order{1}, ::AbstractBSpline{Linear}, (ξ,)::Tuple{V}) where {V} = V((-1,1))
@inline _values(::Order, ::AbstractBSpline{Linear}, (ξ,)::Tuple{V}) where {V} = V((0,0))
@generated function Base.values(::Order{k}, spline::AbstractBSpline{Linear}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x)
        ξ = @. x′ - T((0,1))
        @ntuple $(k+1) a -> _values(Order(a-1), spline, (ξ,))
    end
end

# quadratic
@inline _values(::Order{0}, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = @. muladd($V((0.5,-1.0,0.5)), ξ², muladd($V((-1.5,0.0,1.5)), ξ, $V((1.125,0.75,1.125))))
@inline _values(::Order{1}, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = @. muladd($V((1.0,-2.0,1.0)), ξ, $V((-1.5,0.0,1.5)))
@inline _values(::Order{2}, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = V((1.0,-2.0,1.0))
@inline _values(::Order, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = V((0,0,0))
@generated function Base.values(::Order{k}, spline::AbstractBSpline{Quadratic}, x::Real) where {k}
    quote
        T = typeof(x)
        x′ = fract(x - T(0.5))
        ξ = @. x′ - T((-0.5,0.5,1.5))
        ξ² = @. ξ * ξ
        @ntuple $(k+1) a -> _values(Order(a-1), spline, (ξ,ξ²))
    end
end

# cubic
@inline _values(::Order{0}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = @. muladd($V((-1/6,0.5,-0.5,1/6)), ξ³, muladd($V((1,-1,-1,1)), ξ², muladd($V((-2,0,0,2)), ξ, $V((4/3,2/3,2/3,4/3)))))
@inline _values(::Order{1}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = @. muladd($V((-0.5,1.5,-1.5,0.5)), ξ², muladd($V((2,-2,-2,2)), ξ, $V((-2,0,0,2))))
@inline _values(::Order{2}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = @. muladd($V((-1,3,-3,1)), ξ, $V((2,-2,-2,2)))
@inline _values(::Order{3}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = V((-1,3,-3,1))
@inline _values(::Order, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = V((0,0,0,0))
@generated function Base.values(::Order{k}, spline::AbstractBSpline{Cubic}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x)
        ξ = @. x′ - T((-1,0,1,2))
        ξ² = @. ξ * ξ
        ξ³ = @. ξ² * ξ
        @ntuple $(k+1) a -> _values(Order(a-1), spline, (ξ,ξ²,ξ³))
    end
end

# quartic
@inline _values(::Order{0}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1/24,-1/6,1/4,-1/6,1/24)), ξ⁴, muladd($V((-5/12,5/6,0,-5/6,5/12)), ξ³, muladd($V((25/16,-5/4,-5/8,-5/4,25/16)), ξ², muladd($V((-125/48,5/24,0,-5/24,125/48)), ξ, $V((625/384,55/96,115/192,55/96,625/384))))))
@inline _values(::Order{1}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1/6,-2/3,1,-2/3,1/6)), ξ³, muladd($V((-5/4,5/2,0,-5/2,5/4)), ξ², muladd($V((25/8,-5/2,-5/4,-5/2,25/8)), ξ, $V((-125/48,5/24,0,-5/24,125/48)))))
@inline _values(::Order{2}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1/2,-2,3,-2,1/2)), ξ², muladd($V((-5/2,5,0,-5,5/2)), ξ, $V((25/8,-5/2,-5/4,-5/2,25/8))))
@inline _values(::Order{3}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1,-4,6,-4,1)), ξ, $V((-5/2,5,0,-5,5/2)))
@inline _values(::Order{4}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = V((1,-4,6,-4,1))
@inline _values(::Order, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = V((0,0,0,0,0))
@generated function Base.values(::Order{k}, spline::AbstractBSpline{Quartic}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x - T(0.5))
        ξ = @. x′ - T((-1.5,-0.5,0.5,1.5,2.5))
        ξ² = @. ξ * ξ
        ξ³ = @. ξ² * ξ
        ξ⁴ = @. ξ² * ξ²
        @ntuple $(k+1) a -> _values(Order(a-1), spline, (ξ,ξ²,ξ³,ξ⁴))
    end
end

@generated function _values(::Order{0}, vals::Vararg{Tuple, dim}) where {dim}
    quote
        @_inline_meta
        tuple_otimes(@ntuple $dim d -> vals[d][1])
    end
end
@generated function _values(::Order{k}, vals::Vararg{Tuple, dim}) where {k, dim}
    if k == 1
        TT = Vec{dim}
    else
        TT = Tensor{Tuple{@Symmetry{fill(dim,k)...}}}
    end
    v = Array{Expr}(undef, size(TT))
    for I in CartesianIndices(v)
        ex = Expr(:tuple)
        for i in 1:dim
            j = count(==(i), Tuple(I)) + 1
            push!(ex.args, :(vals[$i][$j]))
        end
        v[I] = ex
    end
    quote
        @_inline_meta
        v = $(Expr(:tuple, v[Tensorial.indices_unique(TT)]...))
        map($TT, map(tuple_otimes, v)...)
    end
end

@generated function Base.values(order::Order{k}, spline::AbstractBSpline, x::Vec{dim}) where {k, dim}
    quote
        @_inline_meta
        vals = @ntuple $dim d -> values(order, spline, x[d])
        @ntuple $(k+1) a -> _values(Order(a-1), vals...)
    end
end
@inline tuple_otimes(x::Tuple) = SArray(otimes(map(Vec, x)...))

@generated function Base.values(order::Order{k}, spline::AbstractBSpline, x::Vec, mesh::CartesianMesh) where {k}
    quote
        @_inline_meta
        xmin = get_xmin(mesh)
        h⁻¹ = spacing_inv(mesh)
        vals = values(order, spline, (x-xmin)*h⁻¹)
        @ntuple $(k+1) i -> vals[i]*h⁻¹^(i-1)
    end
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
`degree` is one of `Linear()`, `Quadratic()` or `Cubic()`.

!!! warning
    `BSpline(Quadratic())` and `BSpline(Cubic())` cannot handle boundaries correctly
    because the kernel values are merely truncated, which leads to unstable behavior.
    Therefore, it is recommended to use either `SteffenBSpline` or `KernelCorrection`
    in cases where proper handling of boundaries is necessary.
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
    SteffenBSpline(degree)

B-spline kernel with boundary correction by Steffen et al.[^Steffen]
`SteffenBSpline` satisfies the partition of unity, ``\\sum_i w_{ip} = 1``, near boundaries.
See also [`KernelCorrection`](@ref).

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
