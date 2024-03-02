abstract type AbstractPolynomial end

@inline function value(p::AbstractPolynomial, x::Vec, ::Symbol)
    ∇P, P = gradient(x->value(p,x), x, :all)
    P, ∇P
end

struct Polynomial{order} <: AbstractPolynomial
    function Polynomial{order}() where {order}
        new{order::Int}()
    end
end

const LinearPolynomial = Polynomial{1}
value_length(::LinearPolynomial, x::Vec) = 1 + length(x)
@inline value(::LinearPolynomial, x::Vec) = vcat(one(eltype(x)), x)

struct MultilinearPolynomial <: AbstractPolynomial
end

value_length(::MultilinearPolynomial, ::Vec{1}) = 2
@inline value(::MultilinearPolynomial, x::Vec{1}) = Vec(one(eltype(x)), x[1])

value_length(::MultilinearPolynomial, ::Vec{2}) = 4
@inline value(::MultilinearPolynomial, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2])

value_length(::MultilinearPolynomial, ::Vec{3}) = 8
@inline value(::MultilinearPolynomial, x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]*x[2]*x[3])
