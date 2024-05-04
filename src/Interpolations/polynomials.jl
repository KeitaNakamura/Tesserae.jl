abstract type AbstractPolynomial end

struct Polynomial{order} <: AbstractPolynomial
    function Polynomial{order}() where {order}
        new{order::Int}()
    end
end

const LinearPolynomial = Polynomial{1}
@inline (::LinearPolynomial)(x::Vec) = vcat(one(eltype(x)), x)

const QuadraticPolynomial = Polynomial{2}
@inline (::QuadraticPolynomial)(x::Vec{1}) = Vec(one(eltype(x)), x[1], x[1]^2)
@inline (::QuadraticPolynomial)(x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2], x[1]^2, x[2]^2)
@inline (::QuadraticPolynomial)(x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]^2, x[2]^2, x[3]^2)

struct MultiPolynomial{order} <: AbstractPolynomial
    function MultiPolynomial{order}() where {order}
        new{order::Int}()
    end
end

const MultiLinearPolynomial = MultiPolynomial{1}

@inline (::MultiPolynomial)(x::Vec{1}) = Vec(one(eltype(x)), x[1])
@inline (::MultiPolynomial)(x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2])
@inline (::MultiPolynomial)(x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]*x[2]*x[3])
