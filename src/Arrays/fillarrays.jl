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

struct ScalarMatrix{T} <: AbstractArray{T, 2}
    val::T
    dims::NTuple{2, Int}
end
ScalarMatrix(v, i::Int, j::Int) = ScalarMatrix(v, (i,j))
Base.size(x::ScalarMatrix) = x.dims
value(x::ScalarMatrix) = x.val
@inline function Base.getindex(x::ScalarMatrix, i::Int, j::Int)
    @boundscheck checkbounds(x, i, j)
    val = value(x)
    i == j ? val : zero(val)
end
