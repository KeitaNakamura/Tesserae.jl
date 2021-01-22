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
    Tensor2D,
    Tensor3D

include("ScalVec.jl")
include("VecTensor.jl")
include("utils.jl")

end
