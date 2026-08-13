abstract type AbstractBSpline{D <: Degree} <: Kernel end

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
_maximum_bspline_degree(::BSpline) = 5
Base.show(io::IO, spline::BSpline) = print(io, BSpline, "(", spline.degree, ")")

@inline function support_width(spline::AbstractBSpline{Degree{n}}) where {n}
    max_degree = _maximum_bspline_degree(spline)
    0 ≤ n ≤ max_degree || throw(ArgumentError("$(nameof(typeof(spline))) degree must be between 0 and $max_degree"))
    n + 1
end

# Equivalent to supportnodes(x, (n + 1) / 2, mesh), with one floor per axis.
@inline function supportnodes(::AbstractBSpline{Degree{n}}, pt, mesh::CartesianMesh{dim}) where {n, dim}
    x = getx(pt)
    ξ = Tuple(normalize(x, mesh))
    dims = size(mesh)
    isinside(ξ, dims) || return EmptyCartesianIndices(Val(dim))
    T = eltype(x)
    offset = T(n - 1) / T(2)
    start = @. unsafe_trunc(Int, floor(ξ - offset)) + 1
    stop = @. start + n
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end

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

function _evalpoly_expr(coeffs)
    last_nonzero = findlast(!iszero, coeffs)
    last_nonzero === nothing && return :(zero(x))
    typed_coeffs = map(coeffs[1:last_nonzero]) do c
        denominator(c) == 1 ? :(T($(numerator(c)))) : :(T($(numerator(c))) / T($(denominator(c))))
    end
    :(evalpoly(ξ, ($(typed_coeffs...),)))
end

function _bspline_order_expr(coeffs, order)
    entries = map(coeffs) do node_coeffs
        _evalpoly_expr(_derivative_coeffs(node_coeffs, order))
    end
    Expr(:tuple, entries...)
end

function _bspline_values1d_expr(k::Int, degree::Int)
    coeffs = _bspline_local_coeffs(degree)
    values = map(order -> _bspline_order_expr(coeffs, order), 0:k)
    Expr(:tuple, values...)
end

@generated function bspline_axis_values(::Order{k}, spline::BSpline{Degree{n}}, x::Real) where {k, n}
    0 ≤ n || error("B-spline degree must be non-negative")
    ξ_expr = isodd(n) ? :(fract(x)) : :(fract(x - T(0.5)))
    values_expr = _bspline_values1d_expr(k, n)
    quote
        @_inline_meta
        T = typeof(x)
        ξ = $ξ_expr
        $values_expr
    end
end

@inline function update_basis_values!(bw::BasisWeight, spline::BSpline, pt, mesh::CartesianMesh)
    indices = supportnodes(bw)
    if has_full_support(bw, indices)
        update_bspline_full_support!(bw, derivative_order(bw), spline, getx(pt), mesh)
    else
        update_basis_values_nodewise!(bw, spline, pt, mesh)
    end
end

_bspline_var(r, d) = Symbol(:v, r, :_, d)

_bspline_value_type(::Val{1}, dim) = Vec{dim}
_bspline_value_type(::Val{k}, dim) where {k} = Tensor{Tuple{@Symmetry{fill(dim, k)...}}}

# Build the scalar product for one tensor-product component.
function _bspline_product_expr(terms)
    @assert !isempty(terms)
    length(terms) == 1 && return only(terms)
    Expr(:call, :*, terms...)
end

# A component index such as CartesianIndex(2, 1, 1) means ∂/∂x₁ for dim == 3.
function _bspline_derivative_expr(component::CartesianIndex, dim)
    counts = map(d -> count(==(d), Tuple(component)), 1:dim)
    terms = map(d -> _bspline_var(counts[d], d), 1:dim)
    _bspline_product_expr(terms)
end

# Store only independent components for symmetric derivative tensors.
_bspline_component_indices(::Val{1}, dim) = map(CartesianIndex, 1:dim)
function _bspline_component_indices(::Val{k}, dim) where {k}
    TT = _bspline_value_type(Val(k), dim)
    Array(CartesianIndices(size(TT))[Tensorial.independent_to_component_map(TT)])
end

# Generate one value to write into the BasisWeight property arrays.
function _bspline_value_expr(::Val{0}, dim)
    _bspline_product_expr(map(d -> _bspline_var(0, d), 1:dim))
end
function _bspline_value_expr(::Val{k}, dim) where {k}
    TT = _bspline_value_type(Val(k), dim)
    hpow = Symbol(:hpow_, k)
    entries = map(J -> :($(_bspline_derivative_expr(J, dim)) * $hpow),
                  _bspline_component_indices(Val(k), dim))
    :($TT(($(entries...),)))
end

# Fill the full-support BasisWeight arrays directly, avoiding prod_each_dimension
# and the following copyto! into BasisWeight storage.
@generated function update_bspline_full_support!(bw::BasisWeight, order::Order{k}, spline::BSpline{Degree{n}}, x, mesh::CartesianMesh{dim}) where {k, n, dim}
    dims = ntuple(_ -> n + 1, dim)
    node_assignments = map(enumerate(CartesianIndices(dims))) do (i, I)
        # Load all 1D basis values needed at this support node.
        loads = [:($(_bspline_var(r, d)) = vals1d[$d][$(r + 1)][$(I[d])]) for d in 1:dim for r in 0:k]
        # Write value, gradient, Hessian, ... directly into each stored array.
        assignments = [:($(Symbol(:vals_, r))[$i] = $(_bspline_value_expr(Val(r), dim))) for r in 0:k]
        quote
            $(loads...)
            $(assignments...)
        end
    end

    quote
        @_inline_meta
        xmin = get_xmin(mesh)
        h⁻¹ = spacing_inv(mesh)
        ξ = (x - xmin) * h⁻¹
        vals1d = @ntuple $dim d -> bspline_axis_values(order, spline, ξ[d])
        @nexprs $(k + 1) r -> vals_{r-1} = nodal_basis_values(bw, Order(r-1))
        @nexprs $k r -> hpow_r = h⁻¹^r
        @inbounds begin
            $(node_assignments...)
        end
        bw
    end
end

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

# Covers `SteffenBSpline` too; the two differ only in `axis_jet_args`.
@inline nodal_basis_jet(order::Order, spline::AbstractBSpline, pt, mesh::CartesianMesh, i) =
    separable_nodal_basis_jet(order, spline, pt, mesh, i)

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
_maximum_bspline_degree(::SteffenBSpline) = 3
Base.show(io::IO, spline::SteffenBSpline) = print(io, SteffenBSpline, "(", spline.degree, ")")

@inline function update_basis_values!(bw::BasisWeight, spline::SteffenBSpline, pt, mesh::CartesianMesh)
    indices = supportnodes(bw)
    if spline isa Union{SteffenBSpline{Constant}, SteffenBSpline{Linear}}
        update_basis_values!(bw, BSpline(spline.degree), pt, mesh)
    elseif has_steffen_interior_support(bw, indices, mesh)
        update_bspline_full_support!(bw, derivative_order(bw), BSpline(spline.degree), getx(pt), mesh)
    else
        update_basis_values_nodewise!(bw, spline, pt, mesh)
    end
end
# Steffen differs from the cardinal B-spline only at the two boundary nodes.
@inline function has_steffen_interior_support(bw::BasisWeight, indices::CartesianIndices{dim}, mesh::CartesianMesh{dim}) where {dim}
    has_full_support(bw, indices) || return false
    @inbounds for d in 1:dim
        nodes = indices.indices[d]
        axis = mesh.axes[d]
        first(nodes) - firstindex(axis) > 1 || return false
        lastindex(axis) - last(nodes) > 1 || return false
    end
    true
end

# Steffen, M., Kirby, R. M., & Berzins, M. (2008).
# Analysis and reduction of quadrature errors in the material point method (MPM).
# International journal for numerical methods in engineering, 76(6), 922-948.
function value(::SteffenBSpline{Constant}, ξ::Real, pos::Int)
    value(BSpline(Constant()), ξ)
end
function value(::SteffenBSpline{Linear}, ξ::Real, pos::Int)
    value(BSpline(Linear()), ξ)
end
function value(::SteffenBSpline{Quadratic}, ξ::Real, pos::Int)
    ξ = float(ξ)
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
function value(::SteffenBSpline{Cubic}, ξ::Real, pos::Int)
    ξ = float(ξ)
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

# Steffen's correction needs each axis to know how far its node sits from the boundary.
@inline axis_jet_args(::SteffenBSpline, pt, mesh::CartesianMesh, i) =
    map(tuple, Tuple(steffen_node_position(mesh, i)))

@inline function steffen_node_position(ax::AbstractVector, i::Int)
    left = i - firstindex(ax)
    right = lastindex(ax) - i
    ifelse(left < right, left, -right)
end
steffen_node_position(mesh::CartesianMesh, index::CartesianIndex) = Vec(map(steffen_node_position, mesh.axes, Tuple(index)))
