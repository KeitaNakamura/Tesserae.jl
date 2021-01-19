struct Polynomial{order}
    function Polynomial{order}() where {order}
        new{order::Int}()
    end
end

# value
value(poly::Polynomial{0}, x::Vec) = Vec(one(eltype(x)))
value(poly::Polynomial{1}, x::Vec{1, T}) where {T} = @inbounds Vec(one(T), x[1])
value(poly::Polynomial{1}, x::Vec{2, T}) where {T} = @inbounds Vec(one(T), x[1], x[2])
value(poly::Polynomial{1}, x::Vec{3, T}) where {T} = @inbounds Vec(one(T), x[1], x[2], x[3])
value(p::Polynomial, x::AbstractCollection) = lazy(value, p, x)
# gradient
function gradient(poly::Polynomial{1}, x::Vec{1, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z
          o]
end
function gradient(poly::Polynomial{1}, x::Vec{2, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z z
          o z
          z o]
end
function gradient(poly::Polynomial{1}, x::Vec{3, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z z z
          o z z
          z o z
          z z o]
end
gradient(p::Polynomial, x::AbstractCollection) = lazy(gradient, p, x)

# for ∇ operation
struct PolynomialGradient{order}
    parent::Polynomial{order}
end
∇(p::Polynomial{order}) where {order} = PolynomialGradient(p)

# function like methods
(p::Polynomial)(x) = value(p, x)
(p::PolynomialGradient)(x) = gradient(p.parent, x)
