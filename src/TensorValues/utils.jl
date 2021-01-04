const ∇ₛ = symmetric ∘ ∇

Tensors.:⋅(::typeof(∇), v::VectorTensor) = tr(∇(v))
Tensors.:⋅(v::VectorTensor, ::typeof(∇)) = tr(∇(v))

_otimes_(x::Real, v::Vec) = x * v
_otimes_(v::Vec, x::Real) = v * x

_otimes_(v1::Vec, v2::Vec) = v1 ⊗ v2
