struct MultiLinear    end
struct MultiQuadratic end

struct Polynomial{D <: Union{Degree, MultiLinear, MultiQuadratic}}
    degree::D
end
Base.show(io::IO, poly::Polynomial) = print(io, Polynomial, "(", poly.degree, ")")

@inline value(::Polynomial{Linear}, x::Vec) = vcat(one(eltype(x)), x)

@inline value(::Polynomial{Quadratic}, x::Vec{1}) = Vec(one(eltype(x)), x[1], x[1]^2)
@inline value(::Polynomial{Quadratic}, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2], x[1]^2, x[2]^2)
@inline value(::Polynomial{Quadratic}, x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]^2, x[2]^2, x[3]^2)

@inline value(::Polynomial{MultiLinear}, x::Vec{1}) = Vec(one(eltype(x)), x[1])
@inline value(::Polynomial{MultiLinear}, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2])
@inline value(::Polynomial{MultiLinear}, x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]*x[2]*x[3])

@inline value(::Polynomial{MultiQuadratic}, x::Vec{1}) = Vec(one(eltype(x)), x[1], x[1]^2)
@inline value(::Polynomial{MultiQuadratic}, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2], x[1]^2, x[2]^2, x[1]^2*x[2], x[1]*x[2]^2, x[1]^2*x[2]^2)
@inline value(::Polynomial{MultiQuadratic}, x::Vec{3}) = Vec(one(eltype(x)),
                                                           x[1], x[2], x[3],
                                                           x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]*x[2]*x[3],
                                                           x[1]^2, x[2]^2, x[3]^2,
                                                           x[1]^2*x[2], x[1]^2*x[3], x[1]^2*x[2]*x[3],
                                                           x[2]^2*x[1], x[2]^2*x[3], x[1]*x[2]^2*x[3],
                                                           x[3]^2*x[1], x[3]^2*x[2], x[1]*x[2]*x[3]^2,
                                                           x[1]^2*x[2]^2, x[2]^2*x[3]^2, x[3]^2*x[1]^2,
                                                           x[1]^2*x[2]^2*x[3], x[1]*x[2]^2*x[3]^2, x[1]^2*x[2]*x[3]^2, x[1]^2*x[2]^2*x[3]^2)
