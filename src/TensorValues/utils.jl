const ∇ₛ = symmetric ∘ ∇

Tensors.:⋅(::typeof(∇), v::VectorTensor) = tr(∇(v))
Tensors.:⋅(v::VectorTensor, ::typeof(∇)) = tr(∇(v))
