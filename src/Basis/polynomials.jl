struct MultiLinear end
struct MultiQuadratic end

struct Polynomial{D <: Union{Degree, MultiLinear, MultiQuadratic}}
    degree::D
end
Base.show(io::IO, poly::Polynomial) = print(io, Polynomial, "(", poly.degree, ")")

# Higher degrees are deliberately excluded: a kernel support does not carry enough
# independent nodes to condition the moment matrix, so `inv(M)` silently returns
# garbage rather than failing.
const CorrectionPolynomial = Polynomial{<: Union{Linear, MultiLinear}}

function unsupported_correction_polynomial(name, poly)
    "$name supports Polynomial(Linear()) and Polynomial(MultiLinear()), got $poly. \
     A higher-degree basis leaves the weighted least-squares moment matrix singular \
     on a kernel support, producing negative and unbounded weights."
end

@inline value(p::Polynomial, x::Vec) = _value(Order(0), p, x)
@generated function jet(::Order{k}, p::Polynomial{<: Union{Linear, MultiLinear}}, x::Vec) where {k}
    quote
        @_inline_meta
        @ntuple $(k+1) i -> _value(Order(i-1), p, x)
    end
end

@inline _value(::Order{0}, ::Polynomial{Linear}, x::Vec) = vcat(one(eltype(x)), x)
@inline _value(::Order{1}, ::Polynomial{Linear}, x::Vec{dim, T}) where {dim, T} = vcat(zero(Mat{1, dim, T}), one(Mat{dim, dim, T}))
@inline _value(::Order{k}, ::Polynomial{Linear}, x::Vec{dim, T}) where {k, dim, T} = zero(Tensor{Tuple{dim+1, @Symmetry{nfill(dim,Val(k))...}}, T})
@inline _value(::Order{0}, ::Polynomial{Quadratic}, x::Vec{1}) = Vec(one(eltype(x)), x[1], x[1]^2)
@inline _value(::Order{0}, ::Polynomial{Quadratic}, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2], x[1]^2, x[2]^2)
@inline _value(::Order{0}, ::Polynomial{Quadratic}, x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]^2, x[2]^2, x[3]^2)

@inline _value(::Order{k}, ::Polynomial{MultiLinear}, x::Vec{1}) where {k} = _value(Order(k), Polynomial(Linear()), x)
# Each tuple lists the variables of a monomial, preserving multiplication order.
multilinear_terms(::Val{2}) = ((), (1,), (2,), (1,2))
multilinear_terms(::Val{3}) = ((), (1,), (2,), (3,), (1,2), (2,3), (3,1), (1,2,3))

function monomial_derivative_expr(term, derivatives)
    remaining = collect(term)
    for d in derivatives
        index = findfirst(==(d), remaining)
        index === nothing && return 0
        deleteat!(remaining, index)
    end
    factors = [:(x[$d]) for d in remaining]
    isempty(factors) ? 1 : length(factors) == 1 ? only(factors) : Expr(:call, :*, factors...)
end

@inline _value(order::Order, ::Polynomial{MultiLinear}, x::Union{Vec{2},Vec{3}}) = multilinear_value(order, x)
@generated function multilinear_value(::Order{k}, x::Vec{dim,T}) where {k, dim, T}
    terms = multilinear_terms(Val(dim))
    n = length(terms)
    TT = k == 0 ? Vec{n} : k == 1 ? Mat{n,dim} : Tensor{Tuple{n,@Symmetry{ntuple(_ -> dim, k)...}}}
    indices = CartesianIndices(size(TT))[Tensorial.independent_to_component_map(TT)]
    entries = map(indices) do I
        k == 0 && I[1] == 1 ? :(one(T)) : monomial_derivative_expr(terms[I[1]], Base.tail(Tuple(I)))
    end
    # Values use the promotion of their scalar terms, as Vec(one(T), x..., ...)
    # did; derivative tensors retain the explicitly requested element type.
    result_type = k == 0 ? TT : TT{T}
    quote
        @_inline_meta
        $result_type($(entries...))
    end
end

@inline _value(::Order{0}, ::Polynomial{MultiQuadratic}, x::Vec{1}) = Vec(one(eltype(x)), x[1], x[1]^2)
@inline _value(::Order{0}, ::Polynomial{MultiQuadratic}, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2], x[1]^2, x[2]^2, x[1]^2*x[2], x[1]*x[2]^2, x[1]^2*x[2]^2)
@inline _value(::Order{0}, ::Polynomial{MultiQuadratic}, x::Vec{3}) = Vec(one(eltype(x)),
                                                                          x[1], x[2], x[3],
                                                                          x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]*x[2]*x[3],
                                                                          x[1]^2, x[2]^2, x[3]^2,
                                                                          x[1]^2*x[2], x[1]^2*x[3], x[1]^2*x[2]*x[3],
                                                                          x[2]^2*x[1], x[2]^2*x[3], x[1]*x[2]^2*x[3],
                                                                          x[3]^2*x[1], x[3]^2*x[2], x[1]*x[2]*x[3]^2,
                                                                          x[1]^2*x[2]^2, x[2]^2*x[3]^2, x[3]^2*x[1]^2,
                                                                          x[1]^2*x[2]^2*x[3], x[1]*x[2]^2*x[3]^2, x[1]^2*x[2]*x[3]^2, x[1]^2*x[2]^2*x[3]^2)
