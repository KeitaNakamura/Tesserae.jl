"""
    Collection(x, [Val(rank)])
"""
struct Collection{rank, V} <: AbstractCollection{rank}
    parent::V
end

# constructors
Collection{rank}(v::V) where {rank, V} = Collection{rank, V}(v)
Collection(v) = Collection{1}(v)

Base.parent(c::Collection) = c.parent

# needs to be implemented for AbstractCollection
Base.length(c::Collection) = length(parent(c))
@inline function Base.getindex(c::Collection, i::Integer)
    @boundscheck checkbounds(c, i)
    @inbounds parent(c)[i]
end
@inline function Base.setindex!(c::Collection, v, i::Integer)
    @boundscheck checkbounds(c, i)
    @inbounds parent(c)[i] = v
    c
end
