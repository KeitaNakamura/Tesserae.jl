module Arrays

using Reexport

using LinearAlgebra
@reexport using Jams.TensorValues

import SparseArrays: sparse

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
