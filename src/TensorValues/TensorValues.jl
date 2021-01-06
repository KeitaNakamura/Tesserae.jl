module TensorValues

using Reexport
@reexport using Tensors

using Base: @_propagate_inbounds_meta

export ScalarVector, VectorTensor, valgrad, ∇, ∇ₛ, _otimes_

include("value_gradient.jl")
include("utils.jl")

end
