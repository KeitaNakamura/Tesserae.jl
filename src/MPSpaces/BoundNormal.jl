struct BoundNormal{dim, T} <: AbstractArray{Vec{dim, T}, dim}
    dims::NTuple{dim, Int}
end
BoundNormal(::Type{T}, dims::Vararg{Int, dim}) where {dim, T} = BoundNormal{dim, T}(dims)

Base.size(x::BoundNormal) = x.dims

# helper function: check if index is on bound
function onbound(dims::NTuple{dim, Int}, I::CartesianIndex{dim}) where {dim}
    for i in 1:dim
        @inbounds (I[i] == 1 || I[i] == dims[i]) && return true
    end
    false
end
onbound(A::AbstractArray, I::CartesianIndex) = onbound(size(A), I)
onbound(A::AbstractArray, I::Int...) = onbound(size(A), CartesianIndex(I))
onbound(len::Int, i::Int) = i == 1 || i == len

@inline function Base.getindex(x::BoundNormal{dim, T}, I::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(x, I...)
    onbound(x, I...) || return zero(Vec{dim, T})
    v = Vec{dim, T}() do i
        @inbounds begin
            I[i] == 1          ? -one(T) :
            I[i] == size(x, i) ?  one(T) : zero(T)
        end
    end
    v / norm(v)
end
