module Arrays

using Reexport

using LinearAlgebra
@reexport using Tensors

import SparseArrays: sparse
import Jams.TensorValues: ∇, _otimes_, valgrad

using Base: @_propagate_inbounds_meta
using Base.Broadcast: broadcasted
using LinearAlgebra: Adjoint

# sparse arrays

export SparseMatrixCOO
export sparse

# Collections

export AbstractCollection, Collection, LazyCollection
export collection, changerank

include("sparse.jl")
include("collection.jl")

end
