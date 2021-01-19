module TensorValues

using Reexport
@reexport using Tensorial

using Base: @_propagate_inbounds_meta

export
# ScalVec
    ScalVec,
    scalvec,
# VecTensor
    VecTensor,
    vectensor,
# common operations
    valgrad,
    ∇,
# utils
    ∇ₛ,
    symmetric,
    _otimes_,
    tensor2x2,
    tensor3x3

include("ScalVec.jl")
include("VecTensor.jl")
include("utils.jl")

end
