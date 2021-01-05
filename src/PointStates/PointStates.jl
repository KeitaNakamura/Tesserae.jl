module PointStates

using Jams.Arrays: AbstractCollection, UnionCollection, LazyCollection, lazy
using Base: @_propagate_inbounds_meta
using Base.Broadcast: broadcasted

export PointState, ←


struct PointState{T} <: AbstractCollection{2, T}
    data::Vector{T}
end

PointState(::Type{T}, length) where {T} = PointState(Vector{T}(undef, length))
PointState(c::UnionCollection{2}) = (p = PointState(eltype(c), length(c)); p ← c)

Base.length(p::PointState) = length(p.data)

@inline function Base.getindex(p::PointState, i::Int)
    @boundscheck checkbounds(p, i)
    @inbounds p.data[i]
end

@inline function Base.setindex!(p::PointState, v, i::Int)
    @boundscheck checkbounds(p, i)
    @inbounds p.data[i] = v
end

Base.fill!(p::PointState, v) = fill!(p.data, v)

Base.similar(p::PointState, ::Type{T}) where {T} = PointState(T, length(p))
Base.similar(p::PointState{T}) where {T} = similar(p, T)


# left arrow

set!(p::PointState, c::UnionCollection{2}) = (p.data .= c; p)
const ← = set!

end
