# override Tensors.symmetric because it's slow
function symmetric(A::Tensor{2, 2})
    @inbounds SymmetricTensor{2, 2}((A[1,1], (A[2,1]+A[1,2])/2, A[2,2]))
end
function symmetric(A::Tensor{2, 3})
    @inbounds SymmetricTensor{2, 3}((A[1,1], (A[2,1]+A[1,2])/2, (A[3,1]+A[1,3])/2, A[2,2], (A[3,2]+A[2,3])/2, A[3,3]))
end

const ∇ₛ = symmetric ∘ ∇

Tensors.:⋅(::typeof(∇), v::VecTensor) = tr(∇(v))
Tensors.:⋅(v::VecTensor, ::typeof(∇)) = tr(∇(v))

_otimes_(x::Real, v::Vec) = x * v
_otimes_(v::Vec, x::Real) = v * x

_otimes_(v1::Vec, v2::Vec) = v1 ⊗ v2

function tensor3x3(x::Tensor{2,2,T}) where {T}
    z = zero(T)
    @inbounds Tensor{2,3,T}((x[1,1], x[2,1], z, x[1,2], x[2,2], z, z, z, z))
end

function tensor3x3(x::SymmetricTensor{2,2,T}) where {T}
    z = zero(T)
    @inbounds SymmetricTensor{2,3,T}((x[1,1], x[2,1], z, x[2,2], z, z))
end

function tensor2x2(x::Tensor{2,3,T}) where {T}
    @inbounds Tensor{2,2,T}((x[1,1], x[2,1], x[2,1], x[2,2]))
end

function tensor2x2(x::SymmetricTensor{2,3,T}) where {T}
    @inbounds SymmetricTensor{2,2,T}((x[1,1], x[2,1], x[2,2]))
end
