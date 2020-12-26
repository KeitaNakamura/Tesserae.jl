module TensorValues

using Reexport
@reexport using Tensors

using Base: @_propagate_inbounds_meta

export ScalarVector, VectorTensor, ∇, ∇ₛ

include("value_gradient.jl")
include("utils.jl")

end
