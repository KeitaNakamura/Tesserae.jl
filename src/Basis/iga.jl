"""
    IGABasis(degrees)

Basis metadata for tensor-product IGA patches.
"""
struct IGABasis{dim, Degrees <: NTuple{dim, Degree}} <: Basis
    degrees::Degrees
end

degrees(basis::IGABasis) = basis.degrees
igabasis(mesh::IGAMesh) = IGABasis(degrees(first(patches(mesh))))
nsupportnodes(basis::IGABasis) = _nsupportnodes(degrees(basis))

initial_supportnodes(basis::IGABasis, mesh::IGAMesh) = zero(SVector{nsupportnodes(basis), Int})

function allocate_static_basis_values(::Type{Vec{dim, T}}, basis::IGABasis; kwargs...) where {dim, T}
    A = MArray{Tuple{nsupportnodes(basis)}}
    _allocate_basis_values(A, Vec{dim, T}; kwargs...)
end

generate_basis_weights(::Type{T}, mesh::IGAMesh, dims...; kwargs...) where {T} = _generate_basis_weights(T, igabasis(mesh), mesh, _todims(dims...); kwargs...)
generate_basis_weights(mesh::IGAMesh, dims...; kwargs...) = generate_basis_weights(Float64, mesh, dims...; kwargs...)

quadrature_rule(basis::IGABasis) = quadrature_rule(Float64, basis)

function quadrature_rule(::Type{T}, basis::IGABasis{dim}) where {T, dim}
    rules = map(degree -> gauss_legendre_rule(T, degree), degrees(basis))
    qpts1d = map(first, rules)
    qwts1d = map(last, rules)
    indices = CartesianIndices(map(length, qpts1d))
    qpts = vec(map(I -> Vec(map(getindex, qpts1d, Tuple(I))), indices))
    qwts = vec(map(I -> prod(map(getindex, qwts1d, Tuple(I))), indices))
    QuadratureRule(qpts, qwts)
end

function gauss_legendre_rule(::Type{T}, ::Degree{0}) where {T}
    SVector{1, T}((0,)), SVector{1, T}((2,))
end
function gauss_legendre_rule(::Type{T}, ::Degree{1}) where {T}
    x = sqrt(T(1) / 3)
    SVector{2, T}((-x, x)), SVector{2, T}((1, 1))
end
function gauss_legendre_rule(::Type{T}, ::Degree{2}) where {T}
    x = sqrt(T(3) / 5)
    SVector{3, T}((-x, 0, x)), SVector{3, T}((T(5)/9, T(8)/9, T(5)/9))
end
function gauss_legendre_rule(::Type{T}, ::Degree{3}) where {T}
    x1 = sqrt(T(3)/7 - T(2)/7 * sqrt(T(6)/5))
    x2 = sqrt(T(3)/7 + T(2)/7 * sqrt(T(6)/5))
    w1 = (18 + sqrt(T(30))) / 36
    w2 = (18 - sqrt(T(30))) / 36
    SVector{4, T}((-x2, -x1, x1, x2)), SVector{4, T}((w2, w1, w1, w2))
end
function gauss_legendre_rule(::Type{T}, ::Degree{4}) where {T}
    x1 = sqrt(T(5) - 2sqrt(T(10)/7)) / 3
    x2 = sqrt(T(5) + 2sqrt(T(10)/7)) / 3
    w1 = (322 + 13sqrt(T(70))) / 900
    w2 = (322 - 13sqrt(T(70))) / 900
    SVector{5, T}((-x2, -x1, 0, x1, x2)), SVector{5, T}((w2, w1, T(128)/225, w1, w2))
end
function gauss_legendre_rule(::Type{T}, ::Degree{5}) where {T}
    x1 = T(0.2386191860831969)
    x2 = T(0.6612093864662645)
    x3 = T(0.9324695142031521)
    w1 = T(0.4679139345726910)
    w2 = T(0.3607615730481386)
    w3 = T(0.1713244923791704)
    SVector{6, T}((-x3, -x2, -x1, x1, x2, x3)), SVector{6, T}((w3, w2, w1, w1, w2, w3))
end
gauss_legendre_rule(::Type{T}, degree::Degree) where {T} = throw(ArgumentError("IGA quadrature is implemented up to quintic degree"))

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
