module Arrays

using Reexport

using LinearAlgebra, SparseArrays
@reexport using Tensors

import Jams.TensorValues: ∇

using Base: @_propagate_inbounds_meta
using Base.Broadcast: broadcasted
using LinearAlgebra: Adjoint

# SparseMatrixCOO

export SparseMatrixCOO
export sparse

# Collections

export AbstractCollection, Collection
export collection

include("sparse_coo.jl")
include("collection.jl")

end
