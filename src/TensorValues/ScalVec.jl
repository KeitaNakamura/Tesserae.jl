struct ScalVec{dim, T} <: Real
    x::T
    ∇x::Vec{dim, T}
end

scalvec(x::Real, ∇x::Vec) = ScalVec(x, ∇x)

∇(a::ScalVec) = a.∇x

Base.promote(a::ScalVec, x) = promote(a.x, x)
Base.promote(x, a::ScalVec) = promote(x, a.x)

Base.promote_type(::Type{ScalVec{dim, T}}, ::Type{U}) where {dim, T, U} = promote_type(T, U)
Base.promote_type(::Type{U}, ::Type{ScalVec{dim, T}}) where {dim, T, U} = promote_type(U, T)

Base.convert(::Type{T}, a::ScalVec) where {T <: Real} = convert(T, a.x)
Base.convert(::Type{ScalVec{dim, T}}, a::ScalVec) where {dim, T} = ScalVec{dim, T}(a.x, a.∇x)

Base.zero(::Type{ScalVec{dim, T}}) where {dim, T} = ScalVec(zero(T), zero(Vec{dim, T}))
Base.zero(::ScalVec{dim, T}) where {dim, T} = zero(ScalVec{dim, T})

valgrad(x::Real, ∇x::Vec) = ScalVec(x, ∇x)

# scalar vs scalar
for op in (:+, :-, :/, :*)
    @eval Base.$op(a::ScalVec, b::ScalVec) = $op(a.x, b.x)
end

Base.show(io::IO, a::ScalVec) = show(io, a.x)
