struct CollectionView{rank, C <: AbstractCollection{rank}, I} <: AbstractCollection{rank}
    parent::C
    indices::I
end

@inline function Base.view(c::AbstractCollection, I::Vararg{Any, N}) where {N}
    @boundscheck checkbounds(c, I...)
    CollectionView(c, I...)
end

Base.parent(c::AbstractCollection) = c.parent
Base.parentindices(c::AbstractCollection) = c.indices

Base.length(c::CollectionView) = length(parentindices(c))
Base.getindex(c::CollectionView, i::Int) = (@_propagate_inbounds_meta; parent(c)[parentindices(c)[i]])
