abstract type RigidBody{dim, T} <: AbstractVector{Vec{dim, T}} end

coordinates(x::RigidBody) = x.coordinates

Base.size(x::RigidBody) = size(coordinates(x))
@inline function Base.getindex(x::RigidBody, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds coordinates(x)[i]
end
@inline function Base.setindex!(x::RigidBody, v, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds coordinates(x)[i] = v
    x
end
