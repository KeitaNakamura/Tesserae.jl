abstract type AbstractBSpline{D <: Degree} <: Kernel end

kernel_support(::AbstractBSpline{Degree{0}}) = 1
kernel_support(::AbstractBSpline{Degree{1}}) = 2
kernel_support(::AbstractBSpline{Degree{2}}) = 3
kernel_support(::AbstractBSpline{Degree{3}}) = 4
kernel_support(::AbstractBSpline{Degree{4}}) = 5
kernel_support(::AbstractBSpline{Degree{5}}) = 6

@inline function supportnodes(spline::AbstractBSpline, pt, mesh::CartesianMesh{dim}) where {dim}
    x = getx(pt)
    ξ = Tuple(normalize(x, mesh))
    dims = size(mesh)
    isinside(ξ, dims) || return EmptyCartesianIndices(Val(dim))
    offset = _supportnodes_offset(eltype(x), spline)
    r = kernel_support(spline) - 1
    start = @. unsafe_trunc(Int, floor(ξ - offset)) + 1
    stop = @. start + r
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end
@inline _supportnodes_offset(::Type{T}, ::AbstractBSpline{Degree{0}}) where {T} = T(-0.5)
@inline _supportnodes_offset(::Type{T}, ::AbstractBSpline{Degree{1}}) where {T} = T(0.0)
@inline _supportnodes_offset(::Type{T}, ::AbstractBSpline{Degree{2}}) where {T} = T(0.5)
@inline _supportnodes_offset(::Type{T}, ::AbstractBSpline{Degree{3}}) where {T} = T(1.0)
@inline _supportnodes_offset(::Type{T}, ::AbstractBSpline{Degree{4}}) where {T} = T(1.5)
@inline _supportnodes_offset(::Type{T}, ::AbstractBSpline{Degree{5}}) where {T} = T(2.0)

@inline value(spline::AbstractBSpline, pt, mesh::CartesianMesh, i) = only(values(Order(0), spline, pt, mesh, i))

@inline fract(x) = x - floor(x)
# Fast calculations for value, gradient and hessian
# `x` must be normalized by `h`

function _bspline_local_coeffs(degree::Int)
    scale = 1 // factorial(degree)
    # A degree-n B-spline has n + 1 local support nodes.
    map(0:degree) do node
        # Coefficients are stored in ascending powers for evalpoly.
        map(0:degree) do power
            binomial(degree, power) * scale * sum(0:(degree - node)) do shift
                (-1)^shift * binomial(degree + 1, shift) *
                (degree - node - shift)^(degree - power)
            end
        end
    end
end

function _derivative_coeffs(coeffs, order)
    degree = length(coeffs) - 1
    order == 0 && return coeffs
    order > degree && return [0//1]
    map(0:(degree - order)) do power
        factor = factorial(power + order) // factorial(power)
        coeffs[power + order + 1] * factor
    end
end

function _right_coeffs(coeffs)
    degree = length(coeffs) - 1
    map(0:degree) do power
        (-1)^power * sum(p -> coeffs[p + 1] * binomial(p, power), power:degree)
    end
end

function _evalpoly_expr(coeffs, var)
    last_nonzero = findlast(!iszero, coeffs)
    last_nonzero === nothing && return :(zero(x))
    typed_coeffs = map(coeffs[1:last_nonzero]) do c
        denominator(c) == 1 ? :(T($(numerator(c)))) : :(T($(numerator(c))) / T($(denominator(c))))
    end
    :(evalpoly($var, ($(typed_coeffs...),)))
end

function _bspline_node_expr(coeffs, node, degree)
    if 2node ≥ degree
        return _evalpoly_expr(coeffs, :left)
    else
        return _evalpoly_expr(_right_coeffs(coeffs), :right)
    end
end

function _bspline_order_expr(coeffs, order, degree)
    entries = map(0:degree, coeffs) do node, node_coeffs
        _bspline_node_expr(_derivative_coeffs(node_coeffs, order), node, degree)
    end
    Expr(:tuple, entries...)
end

function _bspline_values1d_expr(k::Int, degree::Int)
    coeffs = _bspline_local_coeffs(degree)
    values = map(order -> _bspline_order_expr(coeffs, order, degree), 0:k)
    Expr(:tuple, values...)
end

@generated function values1d(::Order{k}, spline::AbstractBSpline{Degree{n}}, x::Real) where {k, n}
    0 ≤ n || error("B-spline degree must be non-negative")
    left_expr = isodd(n) ? :(fract(x)) : :(fract(x - T(0.5)))
    values_expr = _bspline_values1d_expr(k, n)
    quote
        @_inline_meta
        T = typeof(x)
        left = $left_expr
        right = one(x) - left
        $values_expr
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

@inline function update_property!(bw::BasisWeight, spline::AbstractBSpline, pt, mesh::CartesianMesh)
    indices = supportnodes(bw)
    if has_full_support(bw, indices)
        update_property_full!(bw, spline, pt, mesh)
    else
        update_property_truncated!(bw, spline, pt, mesh)
    end
end

function update_property_truncated!(bw::BasisWeight, spline::AbstractBSpline, pt, mesh)
    indices = supportnodes(bw)
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        set_values!(bw, ip, values(derivative_order(bw), spline, getx(pt), mesh, i))
    end
end
@inline function update_property_full!(bw::BasisWeight, spline::AbstractBSpline, pt, mesh)
    set_values!(bw, values(derivative_order(bw), spline, getx(pt), mesh))
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

@inline function value(::BSpline{Constant}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? one(ξ) : zero(ξ)
end
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
    reverse(∂{k}(ξ -> value(spline, ξ), ξ, :all))
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
function value(::SteffenBSpline{Constant}, ξ::Real, pos::Int)
    value(BSpline(Constant()), ξ)
end
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
    reverse(∂{k}(ξ -> value(spline, ξ, pos), ξ, :all))
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
