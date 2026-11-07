# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

@inline function fastsum(f, iter)
    ret = zero(Base._return_type(f, Tuple{eltype(iter)}))
    @simd for x in iter
        ret += @inline f(x)
    end
    ret
end

getx(x) = getproperty(x, first(propertynames(x)))
getx(x::Vec) = x

# A scalar component axis precedes the logical array axes. A one-component
# tensor has the scalar's size, so reinterpret alone does not add that axis.
scalar_components(A::AbstractArray{<:Number}) = reshape(A, 1, size(A)...)
function scalar_components(A::AbstractArray{T}) where {T <: Tensor}
    flat = reinterpret(reshape, eltype(T), A)
    sizeof(T) == sizeof(eltype(T)) ? reshape(flat, 1, size(A)...) : flat
end

struct MapArray{T, N, F, Args <: Tuple} <: AbstractArray{T, N}
    f::F
    args::Args
    function MapArray{T, N, F, Args}(f::F, args::Args) where {T, N, F, Args}
        @assert all(x->size(x)==size(first(args)), args)
        new{T, N, F, Args}(f, args)
    end
end
function maparray(f::F, args...) where {F}
    Args = map(typeof, args)
    A = Base._return_type(map, Tuple{F, Args...})
    MapArray{eltype(A), ndims(A), F, Tuple{Args...}}(f, args)
end
function maparray(f::Type{T}, args...) where {T}
    Args = map(typeof, args)
    MapArray{T, ndims(first(args)), Type{T}, Tuple{Args...}}(T, args)
end
Base.size(A::MapArray) = size(first(A.args))
Base.IndexStyle(::Type{<: MapArray}) = IndexCartesian()
@inline function Base.getindex(A::MapArray, i::Integer...)
    @boundscheck checkbounds(A, i...)
    @inbounds A.f(getindex.(A.args, i...)...)
end

struct Trues{N} <: AbstractArray{Bool, N}
    dims::Dims{N}
end
Base.size(t::Trues) = t.dims
Base.IndexStyle(::Type{<: Trues}) = IndexLinear()
@inline function Base.getindex(t::Trues, i::Integer)
    @boundscheck checkbounds(t, i)
    true
end

@inline sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
@inline sub2ind(::Dims, ::Nothing)::Int = 0
@inline sub2ind(lin::LinearIndices, I::CartesianIndex)::Int = @inbounds lin[I]
@inline sub2ind(::LinearIndices, ::Nothing)::Int = 0

@generated function dropat(entries::Tuple{Vararg{Any, N}}, index::Int) where {N}
    branches = map(1:N) do i
        kept = map(j -> :(entries[$j]), filter(!=(i), 1:N))
        :(index == $i && return tuple($(kept...)))
    end
    quote
        $(branches...)
        throw(ArgumentError("index must be between 1 and tuple length"))
    end
end
