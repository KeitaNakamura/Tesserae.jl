struct MultiLinear end
struct MultiQuadratic end
struct AdaptiveLinear end

struct Polynomial{D <: Union{Degree, MultiLinear, MultiQuadratic, AdaptiveLinear}}
    degree::D
end
Base.show(io::IO, poly::Polynomial) = print(io, Polynomial, "(", poly.degree, ")")

@inline value(p::Polynomial, x::Vec) = _value(Order(0), p, x)
@generated function Base.values(::Order{k}, p::Polynomial, x::Vec) where {k}
    quote
        @_inline_meta
        @ntuple $(k+1) i -> _value(Order(i-1), p, x)
    end
end

@inline Base.values(::Order{k}, p::Polynomial{<: Union{Quadratic, MultiQuadratic}}, x::Vec) where {k} = ∂ⁿ{k,:all}(x->value(p,x), x)

@inline _value(::Order{0}, ::Polynomial{Linear}, x::Vec) = vcat(one(eltype(x)), x)
@inline _value(::Order{1}, ::Polynomial{Linear}, x::Vec{dim, T}) where {dim, T} = vcat(zero(Mat{1, dim, T}), one(Mat{dim, dim, T}))
@inline _value(::Order{k}, ::Polynomial{Linear}, x::Vec{dim, T}) where {k, dim, T} = zero(Tensor{Tuple{dim+1, @Symmetry{nfill(dim,Val(k))...}}, T})
@inline _value(::Order{0}, ::Polynomial{Quadratic}, x::Vec{1}) = Vec(one(eltype(x)), x[1], x[1]^2)
@inline _value(::Order{0}, ::Polynomial{Quadratic}, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2], x[1]^2, x[2]^2)
@inline _value(::Order{0}, ::Polynomial{Quadratic}, x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]^2, x[2]^2, x[3]^2)

@inline _value(::Order{k}, ::Polynomial{MultiLinear}, x::Vec{1}) where {k} = _value(Order(k), Polynomial(Linear()), x)
@inline _value(::Order{0}, ::Polynomial{MultiLinear}, x::Vec{2}) = Vec(one(eltype(x)), x[1], x[2], x[1]*x[2])
@inline _value(::Order{1}, ::Polynomial{MultiLinear}, x::Vec{2, T}) where {T} = Mat{4, 2, T}(0,1,0,x[2],0,0,1,x[1])
@inline _value(::Order{2}, ::Polynomial{MultiLinear}, x::Vec{2, T}) where {T} = Tensor{Tuple{4, @Symmetry{2, 2}}, T}(0,0,0,0,0,0,0,1,0,0,0,0)
@inline _value(::Order{k}, ::Polynomial{MultiLinear}, x::Vec{2, T}) where {k, T} = zero(Tensor{Tuple{4, @Symmetry{nfill(2,Val(k))...}}, T})
@inline _value(::Order{0}, ::Polynomial{MultiLinear}, x::Vec{3}) = Vec(one(eltype(x)), x[1], x[2], x[3], x[1]*x[2], x[2]*x[3], x[3]*x[1], x[1]*x[2]*x[3])
@inline _value(::Order{1}, ::Polynomial{MultiLinear}, x::Vec{3, T}) where {T} = Mat{8, 3, T}(0,1,0,0,x[2],0,x[3],x[2]*x[3],0,0,1,0,x[1],x[3],0,x[1]*x[3],0,0,0,1,0,x[2],x[1],x[1]*x[2])
@inline _value(::Order{2}, ::Polynomial{MultiLinear}, x::Vec{3, T}) where {T} = Tensor{Tuple{8, @Symmetry{3, 3}}, T}(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,x[3],0,0,0,0,0,0,1,x[2],0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,x[1],0,0,0,0,0,0,0,0)
@inline _value(::Order{3}, ::Polynomial{MultiLinear}, x::Vec{3, T}) where {T} = Tensor{Tuple{8, @Symmetry{3, 3, 3}}, T}(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
@inline _value(::Order{k}, ::Polynomial{MultiLinear}, x::Vec{3, T}) where {k, T} = zero(Tensor{Tuple{8, @Symmetry{nfill(3,Val(k))...}}, T})
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
