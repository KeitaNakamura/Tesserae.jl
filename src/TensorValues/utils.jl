const ∇ₛ = symmetric ∘ ∇

Tensorial.:⋅(::typeof(∇), v::VecTensor) = tr(∇(v))
Tensorial.:⋅(v::VecTensor, ::typeof(∇)) = tr(∇(v))

_otimes_(x::Real, v::Vec) = x * v
_otimes_(v::Vec, x::Real) = v * x

_otimes_(v1::Vec, v2::Vec) = v1 ⊗ v2

function tensor3x3(x::SecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[1,2], x[2,2], z, z, z, z)
end

function tensor3x3(x::SymmetricSecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SymmetricSecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[2,2], z, z)
end

function tensor2x2(x::SecondOrderTensor{3,T}) where {T}
    @inbounds SecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,1], x[2,2])
end

function tensor2x2(x::SymmetricSecondOrderTensor{3,T}) where {T}
    @inbounds SymmetricSecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,2])
end

