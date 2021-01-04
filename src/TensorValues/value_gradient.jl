struct ScalarVector{T <: Real, dim} <: Real
    x::T
    ∇x::Vec{dim, T}
end

∇(a::ScalarVector) = a.∇x

Base.promote(a::ScalarVector, x) = promote(a.x, x)
Base.promote(x, a::ScalarVector) = promote(x, a.x)

Base.promote_type(::Type{ScalarVector{T, dim}}, ::Type{U}) where {T, dim, U} = promote_type(T, U)
Base.promote_type(::Type{U}, ::Type{ScalarVector{T, dim}}) where {T, dim, U} = promote_type(U, T)

Base.convert(::Type{T}, a::ScalarVector) where {T <: Real} = convert(T, a.x)
Base.convert(::Type{ScalarVector{T, dim}}, a::ScalarVector) where {T, dim} = ScalarVector{T, dim}(a.x, a.∇x)

Base.zero(::Type{<: ScalarVector{T}}) where {T} = zero(T)
Base.zero(x::ScalarVector) = zero(typeof(x))

# scalar vs scalar
for op in (:+, :-, :/, :*)
    @eval Base.$op(a::ScalarVector, b::ScalarVector) = $op(a.x, b.x)
end

Base.show(io::IO, a::ScalarVector) = show(io, a.x)


struct VectorTensor{dim, T, M} <: AbstractVector{T}
    x::Vec{dim, T}
    ∇x::Tensor{2, dim, T, M}
end

∇(v::VectorTensor) = v.∇x

Base.size(v::VectorTensor) = size(v.x)
Base.getindex(v::VectorTensor, i::Int) = (@_propagate_inbounds_meta; v.x[i])

Base.zero(::Type{<: VectorTensor{dim, T}}) where {dim, T} = zero(Vec{dim, T})
Base.zero(x::VectorTensor) = zero(typeof(x))

# vector vs vector
# +, -, ⋅, ⊗, ×
for op in (:+, :-, :⋅, :⊗, :×)
    @eval begin
        Tensors.$op(a::VectorTensor, b::AbstractVector) = $op(a.x, b)
        Tensors.$op(a::AbstractVector, b::VectorTensor) = $op(a, b.x)
        Tensors.$op(a::VectorTensor, b::VectorTensor) = $op(a.x, b.x)
    end
end

# vector vs number
# *, /
for op in (:*, :/)
    @eval Tensors.$op(a::VectorTensor, b::Number) = $op(a.x, b)
    if op != :/
        @eval Tensors.$op(a::Number, b::VectorTensor) = $op(a, b.x)
    end
end

# vector vs matrix
# *, ⋅
for op in (:*, :⋅)
    @eval begin
        Tensors.$op(a::AbstractMatrix, b::VectorTensor) = $op(a, b.x)
        Tensors.$op(a::VectorTensor, b::AbstractMatrix) = $op(a.x, b)
    end
end

for op in (:gradient, :hessian, :divergence, :curl, :laplace)
    @eval Tensors.$op(f, v::VectorTensor, args...) = Tensors.$op(f, v.x, args...)
end

for op in (:norm, )
    @eval Tensors.$op(v::VectorTensor) = $op(v.x)
end


ValueGradient(x::Real, ∇x::Vec) = ScalarVector(x, ∇x)
ValueGradient(x::Vec, ∇x::Tensor{2}) = VectorTensor(x, ∇x)
