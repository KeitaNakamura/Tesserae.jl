abstract type AbstractBasis end

struct PolynomialBasis{order} <: AbstractBasis
    function PolynomialBasis{order}() where {order}
        new{order::Int}()
    end
end

value(poly::PolynomialBasis{1}, x::Vec) = vcat(one(eltype(x)), x)
function Tensorial.gradient(poly::PolynomialBasis{1}, x::Vec{1, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z
          o]
end
function Tensorial.gradient(poly::PolynomialBasis{1}, x::Vec{2, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z z
          o z
          z o]
end
function Tensorial.gradient(poly::PolynomialBasis{1}, x::Vec{3, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z z z
          o z z
          z o z
          z z o]
end


struct BilinearBasis <: AbstractBasis end

value(poly::BilinearBasis, x::Vec{2, T}) where {T} = @inbounds Vec(one(T), x[1], x[2], x[1]*x[2])
function Tensorial.gradient(poly::BilinearBasis, x::Vec{2, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z    z
          o    z
          z    o
          x[2] x[1]]
end
