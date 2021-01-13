struct AdjointCollection{rank, P <: AbstractCollection{rank}} <: AbstractCollection{rank}
    parent::P
end
Base.parent(c::AdjointCollection) = c.parent
LinearAlgebra.adjoint(c::AbstractCollection) = AdjointCollection(c)

# getindex
Base.IndexStyle(::Type{<: AdjointCollection}) = IndexLinear()
Base.length(c::AdjointCollection) = length(parent(c))
Base.size(c::AdjointCollection) = (1, length(c))
@inline Base.getindex(c::AdjointCollection, i::Int) = (@_propagate_inbounds_meta; parent(c)[i])
@inline Base.getindex(c::AdjointCollection, i::CartesianIndex{2}) = (@_propagate_inbounds_meta; @assert i[1] == 1; parent(c)[i[2]])

# Broadcast
Broadcast.broadcastable(c::AdjointCollection) = c
Broadcast.BroadcastStyle(::Type{<: AdjointCollection}) = Broadcast.DefaultArrayStyle{2}()
