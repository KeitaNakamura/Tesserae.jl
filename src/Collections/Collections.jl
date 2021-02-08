module Collections

using Jams.TensorValues
using Jams.MaterialModels

using Base: @_propagate_inbounds_meta
using Base.Broadcast
using Base.Broadcast: Broadcasted, broadcasted
using LinearAlgebra

export
    AbstractCollection,
    Collection,
    LazyCollection,
    collection,
    lazy

include("AbstractCollection.jl")
include("Collection.jl")
include("AdjointCollection.jl")
include("LazyCollection.jl")
include("CollectionView.jl")

end
