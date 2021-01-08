module TensorValues

using Reexport
@reexport using Tensors

using Base: @_propagate_inbounds_meta

export
# value and gradient
    ScalVec,
    VecTensor,
    scalvec,
    vectensor,
    valgrad,
    ∇,
# utils
    symmetric,
    ∇ₛ,
    _otimes_,
    tensor2x2,
    tensor3x3

include("value_gradient.jl")
include("utils.jl")

end
