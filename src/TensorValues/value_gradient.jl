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

# scalar vs scalar
for op in (:+, :-, :/, :*)
    @eval Base.$op(a::ScalVec, b::ScalVec) = $op(a.x, b.x)
end

Base.show(io::IO, a::ScalVec) = show(io, a.x)


struct VecTensor{dim, T, M} <: AbstractVector{T}
    x::Vec{dim, T}
    ∇x::Tensor{2, dim, T, M}
end

VecTensor{dim, T}(x::Vec{dim, <: Any}, ∇x::Tensor{2, dim, <: Any, M}) where {dim, T, M} =
    VecTensor{dim, T, M}(x, ∇x)

vectensor(x::Vec, ∇x::Tensor{2}) = VecTensor(x, ∇x)

∇(v::VecTensor) = v.∇x

Base.size(v::VecTensor) = size(v.x)
Base.getindex(v::VecTensor, i::Int) = (@_propagate_inbounds_meta; v.x[i])

Base.convert(::Type{T}, a::VecTensor) where {T <: Vec} = convert(T, a.x)
Base.convert(::Type{VecTensor{dim, T}}, a::VecTensor) where {dim, T} = VecTensor{dim, T}(a.x, a.∇x)

Base.zero(::Type{VecTensor{dim, T}}) where {dim, T} = VecTensor(zero(Vec{dim, T}), zero(Tensor{2, dim, T}))
Base.zero(::Type{VecTensor{dim, T, M}}) where {dim, T, M} = zero(VecTensor{dim, T})
Base.zero(::VecTensor{dim, T}) where {dim, T} = zero(VecTensor{dim, T})

# vector vs vector
# +, -, ⋅, ⊗, ×
for op in (:+, :-, :⋅, :⊗, :×)
    @eval begin
        Tensors.$op(a::VecTensor, b::AbstractVector) = $op(a.x, b)
        Tensors.$op(a::AbstractVector, b::VecTensor) = $op(a, b.x)
        Tensors.$op(a::VecTensor, b::VecTensor) = $op(a.x, b.x)
    end
end

# vector vs number
# *, /
for op in (:*, :/)
    @eval Tensors.$op(a::VecTensor, b::Number) = $op(a.x, b)
    if op != :/
        @eval Tensors.$op(a::Number, b::VecTensor) = $op(a, b.x)
    end
end

# vector vs matrix
# *, ⋅
for op in (:*, :⋅)
    @eval begin
        Tensors.$op(a::AbstractMatrix, b::VecTensor) = $op(a, b.x)
        Tensors.$op(a::VecTensor, b::AbstractMatrix) = $op(a.x, b)
    end
end

for op in (:gradient, :hessian, :divergence, :curl, :laplace)
    @eval Tensors.$op(f, v::VecTensor, args...) = Tensors.$op(f, v.x, args...)
end

for op in (:norm, )
    @eval Tensors.$op(v::VecTensor) = $op(v.x)
end


valgrad(x::Real, ∇x::Vec) = ScalVec(x, ∇x)
valgrad(x::Vec, ∇x::Tensor{2}) = VecTensor(x, ∇x)
