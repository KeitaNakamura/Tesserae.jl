"""
    IGABasis(degrees)

Basis metadata for tensor-product IGA patches.
"""
struct IGABasis{dim, Degrees <: NTuple{dim, Degree}} <: Basis
    degrees::Degrees
end

degrees(basis::IGABasis) = basis.degrees
basis(mesh::IGAMesh) = IGABasis(degrees(first(patches(mesh))))
nsupportnodes(basis::IGABasis) = prod(degree -> _degree(degree) + 1, degrees(basis))

initial_supportnodes(basis::IGABasis, mesh::IGAMesh) = zero(SVector{nsupportnodes(basis), Int})

_generate_supportnodes(::IGABasis, mesh::IGAMesh, dims::Dims{2}) = _generate_cell_supportnodes(mesh, dims)
_generate_supportnodes(::IGABasis, ::IGAMesh, ::Dims) = throw(DimensionMismatch("IGA basis weights must have dimensions (quadrature points, cells)"))

function allocate_static_basis_values(::Type{Vec{dim, T}}, basis::IGABasis; kwargs...) where {dim, T}
    A = MArray{Tuple{nsupportnodes(basis)}}
    _allocate_basis_values(A, Vec{dim, T}; kwargs...)
end

generate_basis_weights(::Type{T}, mesh::IGAMesh, dims...; kwargs...) where {T} = _generate_basis_weights(T, basis(mesh), mesh, _todims(dims...); kwargs...)
generate_basis_weights(mesh::IGAMesh, dims...; kwargs...) = generate_basis_weights(Float64, mesh, dims...; kwargs...)

"""
    generate_quadrature_rule(basis::IGABasis)
    generate_quadrature_rule(T, basis::IGABasis)

Generate the standard [`QuadratureRule`](@ref) for `basis`, using `p + 1`
Gauss–Legendre points in each parametric direction of degree `p`. `T` may be
`Float16`, `Float32`, or `Float64` and defaults to `Float64`.
"""
generate_quadrature_rule(basis::IGABasis) = generate_quadrature_rule(Float64, basis)
function generate_quadrature_rule(::Type{T}, basis::IGABasis{dim}) where {T, dim}
    _check_quadrature_float(T)
    rules = map(degree -> _gauss_legendre_rule(T, degree), degrees(basis))
    _tensor_product_quadrature_rule(_tensor_product_family(Val(dim)), rules)
end

# Map parent Gauss points from [-1, 1]^dim to the selected knot span.
span_point(patch::IGAPatch, span::CartesianIndex, X::Vec) = Vec(map(span_point, patch.knot_vectors, Tuple(span), Tuple(X)))
span_point(knot_vector, span, X) = (knot_vector[span] + knot_vector[span+1] + (knot_vector[span+1] - knot_vector[span]) * X) / 2

# Apply only the parent-to-span Jacobian to a Gauss weight; physical Jacobians are handled later.
span_weight(patch::IGAPatch, span::CartesianIndex, w) = w * prod(map(span_weight, patch.knot_vectors, Tuple(span)))
span_weight(knot_vector, span) = (knot_vector[span+1] - knot_vector[span]) / 2

"""
    iga_basis_values_and_gradients(patch::IGAPatch, span::CartesianIndex, ξ::Vec)

Evaluate the tensor-product B-spline basis functions and their parametric
gradients on a patch span.
"""
iga_basis_values_and_gradients(patch::IGAPatch, span::CartesianIndex, ξ::Vec) = _iga_basis_values_and_gradients(degrees(patch), patch.knot_vectors, span, ξ)

@generated function _iga_basis_values_and_gradients(degrees::Degrees, knot_vectors::NTuple{dim, <: AbstractVector}, span::CartesianIndex{dim}, ξ::Vec{dim}) where {dim, Degrees <: NTuple{dim, Degree}}
    degree_types = fieldtypes(Degrees)
    p = map(degree -> degree.parameters[1], degree_types)
    support_dims = map(degree -> degree + 1, p)
    support_indices = map(Tuple, CartesianIndices(support_dims))
    basis = map(i -> Symbol(:basis, i), eachindex(support_indices))
    basis_values = map(zip(basis, support_indices)) do (basis, I)
        :($basis = map(tuple, map(getindex, N, $I), map(getindex, dN, $I)))
    end
    values = map(basis -> :(only(prod_each_dimension(Order(0), $basis...))), basis)
    gradients = map(basis -> :(only(prod_each_dimension(Order(1), $basis...))), basis)
    quote
        values_1d = map(cox_de_boor_values_and_derivatives, degrees, knot_vectors, Tuple(span), Tuple(ξ))
        N = map(first, values_1d)
        dN = map(last, values_1d)
        $(basis_values...)
        SVector{$(prod(support_dims))}($(values...)), SVector{$(prod(support_dims))}($(gradients...))
    end
end

"""
    rational_basis_values_and_gradients(N::SVector{L}, dN::SVector{L}, weights::SVector{L})

Convert B-spline basis values and gradients to rational basis values and
gradients using control point weights.
"""
@inline function rational_basis_values_and_gradients(N::SVector{L}, dN::SVector{L}, weights::SVector{L}) where {L}
    W = sum(N .* weights)
    dW = sum(dN .* weights)
    R = map((Nᵢ, wᵢ) -> Nᵢ*wᵢ/W, N, weights)
    dR = map((Nᵢ, dNᵢ, wᵢ) -> wᵢ*(dNᵢ*W - Nᵢ*dW)/W^2, N, dN, weights)
    R, dR
end

"""
    cox_de_boor_values(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real)

Evaluate the active 1D B-spline basis functions on a knot span with the
Cox-de Boor recursion.

`span` is the knot-span index for `knot_vector[span] ≤ ξ < knot_vector[span+1]`.
For degree `p`, the returned `SVector{p+1}` is ordered by basis index
`span-p:span`. Repeated knots are valid; zero-denominator recursive terms
contribute zero.
"""
@generated function cox_de_boor_values(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real) where {p}
    vals = (:( cox_de_boor_value(degree, knot_vector, span - $p + $(a - 1), ξ) ) for a in 1:(p+1))
    :(SVector{$(p+1)}($(vals...)))
end

# Same ordering as cox_de_boor_values, but returning dN/dξ.
@generated function cox_de_boor_derivatives(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real) where {p}
    vals = (:( cox_de_boor_derivative(degree, knot_vector, span - $p + $(a - 1), ξ) ) for a in 1:(p+1))
    :(SVector{$(p+1)}($(vals...)))
end

"""
    cox_de_boor_values_and_derivatives(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real)

Evaluate the active 1D B-spline basis functions and their derivatives on a
knot span.

The returned `(N, dN)` uses the same basis-index ordering as
`cox_de_boor_values`. For `p ≥ 1`, the degree `p-1` values are shared between
`N` and `dN`, avoiding the repeated recursive work of separate value and
derivative calls.
"""
@generated function cox_de_boor_values_and_derivatives(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real) where {p}
    if p == 0
        return quote
            N = cox_de_boor_value(degree, knot_vector, span, ξ)
            SVector{1}(N), SVector{1}(zero(ξ))
        end
    end

    degree = :(Degree{$(p-1)}())
    lower = map(a -> Symbol(:N, p-1, "_", a), 1:(p+2))
    lower_values = map(1:(p+2)) do a
        :($(lower[a]) = cox_de_boor_value($degree, knot_vector, span - $p + $(a - 1), ξ))
    end
    values = map(1:(p+1)) do a
        quote
            _cox_de_boor_term(
                ξ - knot_vector[span - $p + $(a - 1)],
                knot_vector[span + $(a - 1)] - knot_vector[span - $p + $(a - 1)],
                $(lower[a]),
            ) + _cox_de_boor_term(
                knot_vector[span + $a] - ξ,
                knot_vector[span + $a] - knot_vector[span - $p + $a],
                $(lower[a+1]),
            )
        end
    end
    derivatives = map(1:(p+1)) do a
        quote
            _cox_de_boor_term(
                $p,
                knot_vector[span + $(a - 1)] - knot_vector[span - $p + $(a - 1)],
                $(lower[a]),
            ) - _cox_de_boor_term(
                $p,
                knot_vector[span + $a] - knot_vector[span - $p + $a],
                $(lower[a+1]),
            )
        end
    end
    quote
        $(lower_values...)
        SVector{$(p+1)}($(values...)), SVector{$(p+1)}($(derivatives...))
    end
end

@inline function cox_de_boor_value(::Degree{0}, knot_vector::AbstractVector, i::Int, ξ::Real)
    ifelse(knot_vector[i] ≤ ξ < knot_vector[i+1], one(ξ), zero(ξ))
end
@inline function cox_de_boor_value(::Degree{p}, knot_vector::AbstractVector, i::Int, ξ::Real) where {p}
    degree = Degree{p-1}()
    left = _cox_de_boor_term(ξ - knot_vector[i], knot_vector[i+p] - knot_vector[i], cox_de_boor_value(degree, knot_vector, i, ξ))
    right = _cox_de_boor_term(knot_vector[i+p+1] - ξ, knot_vector[i+p+1] - knot_vector[i+1], cox_de_boor_value(degree, knot_vector, i+1, ξ))
    left + right
end

@inline cox_de_boor_derivative(::Degree{0}, knot_vector::AbstractVector, i::Int, ξ::Real) = zero(ξ)
@inline function cox_de_boor_derivative(::Degree{p}, knot_vector::AbstractVector, i::Int, ξ::Real) where {p}
    degree = Degree{p-1}()
    left = _cox_de_boor_term(p, knot_vector[i+p] - knot_vector[i], cox_de_boor_value(degree, knot_vector, i, ξ))
    right = _cox_de_boor_term(p, knot_vector[i+p+1] - knot_vector[i+1], cox_de_boor_value(degree, knot_vector, i+1, ξ))
    left - right
end

@inline _cox_de_boor_term(a, b, c) = iszero(b) ? zero(c) : a * c / b
