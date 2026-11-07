# IGA reference evaluation and coordinate mapping

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
    iga_span_basis(patch, span, qpt, weights, indices)

Basis values and parametric gradients at one quadrature point of a knot span,
made rational when the mesh carries control-point `weights`.
"""
@inline function iga_span_basis(patch::IGAPatch, span::CartesianIndex, qpt::Vec, weights, indices)
    ξ = span_point(patch, span, qpt)
    N, dNdξ = iga_basis_values_and_gradients(patch, span, ξ)
    rational_basis_values_and_gradients(N, dNdξ, weights, indices)
end

@inline rational_basis_values_and_gradients(N, dN, ::Nothing, indices) = (N, dN)
@inline rational_basis_values_and_gradients(N, dN, weights::AbstractVector, indices) =
    rational_basis_values_and_gradients(N, dN, weights[indices])

"""
    cox_de_boor_values(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real)

Evaluate the active 1D B-spline basis functions on a knot span with the
Cox-de Boor recursion.

`span` is the knot-span index for `knot_vector[span] ≤ ξ < knot_vector[span+1]`.
For degree `p`, the returned `SVector{p+1}` is ordered by basis index
`span-p:span`. Repeated knots are valid; zero-denominator recursive terms
contribute zero.
"""
@inline function cox_de_boor_values(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real) where {p}
    p == 0 && return SVector(cox_de_boor_value(degree, knot_vector, span, ξ))
    @boundscheck checkbounds(knot_vector, (span-p):(span+p+1))
    knot_vector[span] ≤ ξ < knot_vector[span+1] || return recursive_cox_de_boor_values(degree, knot_vector, span, ξ)
    T = promote_type(eltype(knot_vector), typeof(ξ))
    # Static mutable workspaces require isbits elements. Keep the scalar
    # recursion for other arithmetic, including BigFloat and abstract storage.
    T <: AbstractFloat && isbitstype(T) || return recursive_cox_de_boor_values(degree, knot_vector, span, ξ)
    values = MVector{p+1,T}(undef)
    left, right = MVector{p,T}(undef), MVector{p,T}(undef)
    @inbounds bspline_span_values!(values, left, right, knot_vector, span, p, ξ)
    SVector(values)
end

# Retain scalar recursion for queries outside the requested span, including the
# half-open right endpoint; NURBS deliberately includes its clamped endpoint.
@inline function recursive_cox_de_boor_values(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real) where {p}
    SVector(ntuple(a -> cox_de_boor_value(degree, knot_vector, span-p+a-1, ξ), Val(p+1)))
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
@inline function cox_de_boor_values_and_derivatives(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real) where {p}
    p == 0 && return (SVector(cox_de_boor_value(degree, knot_vector, span, ξ)), SVector(zero(ξ)))
    @boundscheck checkbounds(knot_vector, (span-p):(span+p+1))
    knot_vector[span] ≤ ξ < knot_vector[span+1] || return recursive_cox_de_boor_values_and_derivatives(degree, knot_vector, span, ξ)
    T = promote_type(eltype(knot_vector), typeof(ξ))
    T <: AbstractFloat && isbitstype(T) || return recursive_cox_de_boor_values_and_derivatives(degree, knot_vector, span, ξ)
    values = MVector{p+1,T}(undef)
    left, right, lower = MVector{p,T}(undef), MVector{p,T}(undef), MVector{p,T}(undef)
    @inbounds bspline_span_values!(values, left, right, knot_vector, span, p, ξ, lower)
    derivatives = ntuple(Val(p+1)) do a
        @inbounds begin
            l = a == 1 ? zero(T) : lower[a-1]
            r = a == p+1 ? zero(T) : lower[a]
            _cox_de_boor_term(p, knot_vector[span+a-1] - knot_vector[span-p+a-1], l) -
                _cox_de_boor_term(p, knot_vector[span+a] - knot_vector[span-p+a], r)
        end
    end
    SVector(values), SVector(derivatives)
end

@inline function recursive_cox_de_boor_values_and_derivatives(degree::Degree{p}, knot_vector::AbstractVector, span::Int, ξ::Real) where {p}
    lower = ntuple(a -> cox_de_boor_value(Degree{p-1}(), knot_vector, span-p+a-1, ξ), Val(p+2))
    jets = ntuple(Val(p+1)) do a
        i = span-p+a-1
        left = knot_vector[i+p] - knot_vector[i]
        right = knot_vector[i+p+1] - knot_vector[i+1]
        N = _cox_de_boor_term(ξ - knot_vector[i], left, lower[a]) +
            _cox_de_boor_term(knot_vector[i+p+1] - ξ, right, lower[a+1])
        dN = _cox_de_boor_term(p, left, lower[a]) - _cox_de_boor_term(p, right, lower[a+1])
        N, dN
    end
    SVector(map(first, jets)), SVector(map(last, jets))
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

@inline _cox_de_boor_term(a, b, c) = iszero(b) ? zero(c) : a * c / b

function cell_point(mesh::IGAMesh, cell, qpt)
    patch = patches(mesh, cell.patch)
    indices = supportnodes(mesh, cell)
    R, _ = iga_span_basis(patch, cell.span, qpt, mesh.weights, indices)
    sum(R .* mesh[indices])
end
