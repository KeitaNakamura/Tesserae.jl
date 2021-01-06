module Arrays

using Reexport

using LinearAlgebra
@reexport using Jams.DofHelpers
@reexport using Tensors

import SparseArrays: sparse, nonzeros, nnz
import Jams.TensorValues: ∇, _otimes_, valgrad
import Jams.DofHelpers: indices

using Base: @_propagate_inbounds_meta
using Base.Broadcast: broadcasted
using LinearAlgebra: Adjoint

# sparse arrays

export SparseArray, SparseMatrixCOO
export sparse, nonzeros, zeros!

# Collections

export AbstractCollection, Collection, LazyCollection
export collection, changerank

include("sparse.jl")
include("collection.jl")

end
