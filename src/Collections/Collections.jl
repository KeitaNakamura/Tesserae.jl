module Collections

using Poingr.TensorValues
using Poingr.MaterialModels

using Base: @_propagate_inbounds_meta, @_inline_meta, @pure
using LinearAlgebra

export
# AbstractCollection
    AbstractCollection,
    set!,
    ←,
# Collection
    Collection,
# LazyCollection
    LazyCollection,
    LazyOperationType,
    LazyAddLikeOperator,
    LazyMulLikeOperator,
    lazy

include("utils.jl")
include("AbstractCollection.jl")
include("Collection.jl")
include("AdjointCollection.jl")
include("LazyCollection.jl")
include("CollectionView.jl")

const ← = set!

end
