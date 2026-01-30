abstract type AbstractBSpline{D <: Degree} <: Kernel end

kernel_support(::AbstractBSpline{Degree{1}}) = 2
kernel_support(::AbstractBSpline{Degree{2}}) = 3
kernel_support(::AbstractBSpline{Degree{3}}) = 4
kernel_support(::AbstractBSpline{Degree{4}}) = 5
kernel_support(::AbstractBSpline{Degree{5}}) = 6

@inline function neighboringnodes(spline::AbstractBSpline, pt, mesh::CartesianMesh{dim}) where {dim}
    x = getx(pt)
    ξ = Tuple(normalize(x, mesh))
    dims = size(mesh)
    isinside(ξ, dims) || return EmptyCartesianIndices(Val(dim))
    offset = _neighboringnodes_offset(eltype(x), spline)
    r = kernel_support(spline) - 1
    start = @. unsafe_trunc(Int, floor(ξ - offset)) + 1
    stop = @. start + r
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end
@inline _neighboringnodes_offset(::Type{T}, ::AbstractBSpline{Degree{1}}) where {T} = T(0.0)
@inline _neighboringnodes_offset(::Type{T}, ::AbstractBSpline{Degree{2}}) where {T} = T(0.5)
@inline _neighboringnodes_offset(::Type{T}, ::AbstractBSpline{Degree{3}}) where {T} = T(1.0)
@inline _neighboringnodes_offset(::Type{T}, ::AbstractBSpline{Degree{4}}) where {T} = T(1.5)
@inline _neighboringnodes_offset(::Type{T}, ::AbstractBSpline{Degree{5}}) where {T} = T(2.0)

@inline value(spline::AbstractBSpline, pt, mesh::CartesianMesh, i) = only(values(Order(0), spline, pt, mesh, i))

@inline fract(x) = x - floor(x)
# Fast calculations for value, gradient and hessian
# `x` must be normalized by `h`

# linear
@inline _value1d(::Order{0}, ::AbstractBSpline{Linear}, (ξ,)::Tuple{V}) where {V} = @. muladd($V((-1,1)), ξ, $V((1,1)))
@inline _value1d(::Order{1}, ::AbstractBSpline{Linear}, (ξ,)::Tuple{V}) where {V} = V((-1,1))
@inline _value1d(::Order, ::AbstractBSpline{Linear}, (ξ,)::Tuple{V}) where {V} = V((0,0))
@generated function values1d(::Order{k}, spline::AbstractBSpline{Linear}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x)
        ξ = @. x′ - T((0,1))
        @ntuple $(k+1) a -> _value1d(Order(a-1), spline, (ξ,))
    end
end

# quadratic
@inline _value1d(::Order{0}, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = @. muladd($V((0.5,-1.0,0.5)), ξ², muladd($V((-1.5,0.0,1.5)), ξ, $V((1.125,0.75,1.125))))
@inline _value1d(::Order{1}, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = @. muladd($V((1.0,-2.0,1.0)), ξ, $V((-1.5,0.0,1.5)))
@inline _value1d(::Order{2}, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = V((1.0,-2.0,1.0))
@inline _value1d(::Order, ::AbstractBSpline{Quadratic}, (ξ,ξ²)::NTuple{2,V}) where {V} = V((0,0,0))
@generated function values1d(::Order{k}, spline::AbstractBSpline{Quadratic}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x - T(0.5))
        ξ = @. x′ - T((-0.5,0.5,1.5))
        ξ² = @. ξ * ξ
        @ntuple $(k+1) a -> _value1d(Order(a-1), spline, (ξ,ξ²))
    end
end

# cubic
@inline _value1d(::Order{0}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = @. muladd($V((-1/6,0.5,-0.5,1/6)), ξ³, muladd($V((1,-1,-1,1)), ξ², muladd($V((-2,0,0,2)), ξ, $V((4/3,2/3,2/3,4/3)))))
@inline _value1d(::Order{1}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = @. muladd($V((-0.5,1.5,-1.5,0.5)), ξ², muladd($V((2,-2,-2,2)), ξ, $V((-2,0,0,2))))
@inline _value1d(::Order{2}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = @. muladd($V((-1,3,-3,1)), ξ, $V((2,-2,-2,2)))
@inline _value1d(::Order{3}, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = V((-1,3,-3,1))
@inline _value1d(::Order, ::AbstractBSpline{Cubic}, (ξ,ξ²,ξ³)::NTuple{3,V}) where {V} = V((0,0,0,0))
@generated function values1d(::Order{k}, spline::AbstractBSpline{Cubic}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x)
        ξ = @. x′ - T((-1,0,1,2))
        ξ² = @. ξ * ξ
        ξ³ = @. ξ² * ξ
        @ntuple $(k+1) a -> _value1d(Order(a-1), spline, (ξ,ξ²,ξ³))
    end
end

# quartic
@inline _value1d(::Order{0}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1/24,-1/6,1/4,-1/6,1/24)), ξ⁴, muladd($V((-5/12,5/6,0,-5/6,5/12)), ξ³, muladd($V((25/16,-5/4,-5/8,-5/4,25/16)), ξ², muladd($V((-125/48,5/24,0,-5/24,125/48)), ξ, $V((625/384,55/96,115/192,55/96,625/384))))))
@inline _value1d(::Order{1}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1/6,-2/3,1,-2/3,1/6)), ξ³, muladd($V((-5/4,5/2,0,-5/2,5/4)), ξ², muladd($V((25/8,-5/2,-5/4,-5/2,25/8)), ξ, $V((-125/48,5/24,0,-5/24,125/48)))))
@inline _value1d(::Order{2}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1/2,-2,3,-2,1/2)), ξ², muladd($V((-5/2,5,0,-5,5/2)), ξ, $V((25/8,-5/2,-5/4,-5/2,25/8))))
@inline _value1d(::Order{3}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = @. muladd($V((1,-4,6,-4,1)), ξ, $V((-5/2,5,0,-5,5/2)))
@inline _value1d(::Order{4}, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = V((1,-4,6,-4,1))
@inline _value1d(::Order, ::AbstractBSpline{Quartic}, (ξ,ξ²,ξ³,ξ⁴)::NTuple{4,V}) where {V} = V((0,0,0,0,0))
@generated function values1d(::Order{k}, spline::AbstractBSpline{Quartic}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x - T(0.5))
        ξ = @. x′ - T((-1.5,-0.5,0.5,1.5,2.5))
        ξ² = @. ξ * ξ
        ξ³ = @. ξ² * ξ
        ξ⁴ = @. ξ² * ξ²
        @ntuple $(k+1) a -> _value1d(Order(a-1), spline, (ξ,ξ²,ξ³,ξ⁴))
    end
end

# quintic
@inline _value1d(::Order{0}, ::AbstractBSpline{Quintic}, (ξ,ξ²,ξ³,ξ⁴,ξ⁵)::NTuple{5,V}) where {V} = @. muladd($V((-1/120,1/24,-1/12,1/12,-1/24,1/120)), ξ⁵, muladd($V((1/8,-3/8,1/4,1/4,-3/8,1/8)), ξ⁴, muladd($V((-3/4,5/4,0,0,-5/4,3/4)), ξ³, muladd($V((9/4,-7/4,-1/2,-1/2,-7/4,9/4)), ξ², muladd($V((-27/8,5/8,0,0,-5/8,27/8)), ξ, $V((81/40,17/40,11/20,11/20,17/40,81/40)))))))
@inline _value1d(::Order{1}, ::AbstractBSpline{Quintic}, (ξ,ξ²,ξ³,ξ⁴,ξ⁵)::NTuple{5,V}) where {V} = @. muladd($V((-1/24,5/24,-5/12,5/12,-5/24,1/24)), ξ⁴, muladd($V((1/2,-3/2,1,1,-3/2,1/2)), ξ³, muladd($V((-9/4,15/4,0,0,-15/4,9/4)), ξ², muladd($V((9/2,-7/2,-1,-1,-7/2,9/2)), ξ, $V((-27/8,5/8,0,0,-5/8,27/8))))))
@inline _value1d(::Order{2}, ::AbstractBSpline{Quintic}, (ξ,ξ²,ξ³,ξ⁴,ξ⁵)::NTuple{5,V}) where {V} = @. muladd($V((-1/6,5/6,-5/3,5/3,-5/6,1/6)), ξ³, muladd($V((3/2,-9/2,3,3,-9/2,3/2)), ξ², muladd($V((-9/2,15/2,0,0,-15/2,9/2)), ξ, $V((9/2,-7/2,-1,-1,-7/2,9/2)))))
@inline _value1d(::Order{3}, ::AbstractBSpline{Quintic}, (ξ,ξ²,ξ³,ξ⁴,ξ⁵)::NTuple{5,V}) where {V} = @. muladd($V((-1/2,5/2,-5,5,-5/2,1/2)), ξ², muladd($V((3,-9,6,6,-9,3)), ξ, $V((-9/2,15/2,0,0,-15/2,9/2))))
@inline _value1d(::Order{4}, ::AbstractBSpline{Quintic}, (ξ,ξ²,ξ³,ξ⁴,ξ⁵)::NTuple{5,V}) where {V} = @. muladd($V((-1,5,-10,10,-5,1)), ξ, $V((3,-9,6,6,-9,3)))
@inline _value1d(::Order{5}, ::AbstractBSpline{Quintic}, (ξ,ξ²,ξ³,ξ⁴,ξ⁵)::NTuple{5,V}) where {V} = V((-1,5,-10,10,-5,1))
@inline _value1d(::Order, ::AbstractBSpline{Quintic}, (ξ,ξ²,ξ³,ξ⁴,ξ⁵)::NTuple{5,V}) where {V} = V((0,0,0,0,0,0))
@generated function values1d(::Order{k}, spline::AbstractBSpline{Quintic}, x::Real) where {k}
    quote
        @_inline_meta
        T = typeof(x)
        x′ = fract(x)
        ξ = @. x′ - T((-2,-1,0,1,2,3))
        ξ² = @. ξ * ξ
        ξ³ = @. ξ² * ξ
        ξ⁴ = @. ξ² * ξ²
        ξ⁵ = @. ξ² * ξ³
        @ntuple $(k+1) a -> _value1d(Order(a-1), spline, (ξ,ξ²,ξ³,ξ⁴,ξ⁵))
    end
end

@generated function Base.values(order::Order{k}, spline::AbstractBSpline, x::Vec, mesh::CartesianMesh{dim}) where {k, dim}
    quote
        @_inline_meta
        xmin = get_xmin(mesh)
        h⁻¹ = spacing_inv(mesh)
        ξ = (x - xmin) * h⁻¹
        vals1d = @ntuple $dim d -> values1d(order, spline, ξ[d])
        vals = @ntuple $(k+1) a -> prod_each_dimension(Order(a-1), vals1d...)
        @ntuple $(k+1) i -> vals[i]*h⁻¹^(i-1)
    end
end

@inline function update_property!(iw::InterpolationWeight, spline::AbstractBSpline, pt, mesh::CartesianMesh)
    indices = neighboringnodes(iw)
    is_support_truncated = size(values(iw,1)) != size(indices)
    if is_support_truncated
        @inbounds for ip in eachindex(indices)
            i = indices[ip]
            set_values!(iw, ip, values(derivative_order(iw), spline, getx(pt), mesh, i))
        end
    else
        set_values!(iw, values(derivative_order(iw), spline, getx(pt), mesh))
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

@inline function value(::BSpline{Linear}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end
@inline function value(::BSpline{Quadratic}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end
@inline function value(::BSpline{Cubic}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end
@inline function value(::BSpline{Quartic}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (48ξ^4 - 120ξ^2 + 115) / 192              :
    ξ < 1.5 ? -(16ξ^4 - 80ξ^3 + 120ξ^2 - 20ξ - 55) / 96 :
    ξ < 2.5 ? (5 - 2ξ)^4 / 384                          : zero(ξ)
end
@inline function value(::BSpline{Quintic}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? ((3-ξ)^5 - 6*(2-ξ)^5 + 15*(1-ξ)^5) / 120 :
    ξ < 2 ? ((3-ξ)^5 - 6*(2-ξ)^5) / 120              :
    ξ < 3 ? ((3-ξ)^5) / 120                          : zero(ξ)
end

@inline function Base.values(::Order{k}, spline::BSpline, ξ::Real) where {k}
    ∂ⁿ{k,:all}(ξ -> value(spline, ξ), ξ)
end

@generated function Base.values(order::Order{k}, spline::BSpline, pt, mesh::CartesianMesh{dim}, i) where {dim, k}
    quote
        @_inline_meta
        x = getx(pt)
        h⁻¹ = spacing_inv(mesh)
        ξ = (x - mesh[i]) * h⁻¹
        vals1d = @ntuple $dim d -> values(order, spline, ξ[d])
        vals = @ntuple $(k+1) a -> only(prod_each_dimension(Order(a-1), vals1d...))
        @ntuple $(k+1) i -> vals[i]*h⁻¹^(i-1)
    end
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
function value(::SteffenBSpline{Linear}, ξ::Real, pos::Int)
    value(BSpline(Linear()), ξ)
end
function value(::SteffenBSpline{Quadratic}, ξ::Real, pos::Int)::typeof(ξ)
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
        value(BSpline(Quadratic()), ξ)
    end
end
function value(::SteffenBSpline{Cubic}, ξ::Real, pos::Int)::typeof(ξ)
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
        value(BSpline(Cubic()), ξ)
    end
end

@inline function Base.values(::Order{k}, spline::SteffenBSpline, ξ::Real, pos::Int) where {k}
    ∂ⁿ{k,:all}(ξ -> value(spline, ξ, pos), ξ)
end

@generated function Base.values(order::Order{k}, spline::SteffenBSpline, pt, mesh::CartesianMesh{dim}, i) where {dim, k}
    quote
        @_inline_meta
        x = getx(pt)
        h⁻¹ = spacing_inv(mesh)
        ξ = (x - mesh[i]) * h⁻¹
        pos = node_position(mesh, i)
        vals1d = @ntuple $dim d -> values(order, spline, ξ[d], pos[d])
        vals = @ntuple $(k+1) a -> only(prod_each_dimension(Order(a-1), vals1d...))
        @ntuple $(k+1) i -> vals[i]*h⁻¹^(i-1)
    end
end

@inline function node_position(ax::AbstractVector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
node_position(mesh::CartesianMesh, index::CartesianIndex) = Vec(map(node_position, mesh.axes, Tuple(index)))
