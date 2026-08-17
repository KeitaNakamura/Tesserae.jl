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

commas(num::Integer) = replace(string(num), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")

getx(x) = getproperty(x, first(propertynames(x)))
getx(x::Vec) = x
getx(x::Vector{<: Vec}) = x

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
