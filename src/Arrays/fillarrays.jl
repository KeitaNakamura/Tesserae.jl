abstract type AbstractFillArray{T, N} <: AbstractArray{T, N} end

Base.size(x::AbstractFillArray) = x.dims
Base.IndexStyle(::Type{<: AbstractFillArray}) = IndexLinear()
@inline function Base.getindex(x::AbstractFillArray, i::Int)
    @boundscheck checkbounds(x, i)
    value(x)
end

struct FillArray{T, N} <: AbstractFillArray{T, N}
    val::T
    dims::NTuple{N, Int}
end
FillArray(v, dims::Int...) = FillArray(v, dims)
value(x::FillArray) = x.val

for (FT, func) in ((:Ones, one), (:Zeros, zero))
    @eval begin
        struct $FT{T, N} <: AbstractFillArray{T, N}
            dims::NTuple{N, Int}
        end
        $FT{T}(dims::Vararg{Int, N}) where {T, N} = $FT{T, N}(dims)
        $FT{T}(dims::NTuple{N, Int}) where {T, N} = $FT{T, N}(dims)
        @inline value(x::$FT{T}) where {T} = $func(T)
    end
end
