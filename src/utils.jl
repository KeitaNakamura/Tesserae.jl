struct Index{dim}
    i::Int
    I::CartesianIndex{dim}
end
@inline Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::Index) = checkindex(Bool, inds, i.i)
@inline _to_indices(::IndexLinear, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (I[1].i, Base.tail(I)...))
@inline _to_indices(::IndexCartesian, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (Tuple(I[1].I)..., Base.tail(I)...))
@inline Base.to_indices(A, inds, I::Tuple{Index, Vararg{Any}}) = _to_indices(IndexStyle(A), A, inds, I)

nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

promote_tuple_length(xs::Type{<: NTuple{N, Any}}...) where {N} = N
@generated function broadcast_tuple(f, xs::Vararg{Any, N}) where {N}
    L = promote_tuple_length([x for x in xs if x <: Tuple]...)
    exps = map(1:L) do i
        args = [xs[j] <: Tuple ? :(xs[$j][$i]) : :(xs[$j]) for j in 1:N]
        :(f($(args...)))
    end
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        tuple($(exps...))
    end
end

isapproxzero(x::Number) = abs(x) <= sqrt(eps(typeof(x)))

macro _inline_propagate_inbounds_meta()
    quote
        Base.@_inline_meta
        Base.@_propagate_inbounds_meta
    end
end

# elzero
elzero(x) = zero(eltype(x))

# zerorec: `zero` recursively
zerorec(::Type{Array{T, N}}) where {T, N} = Array{T, N}(undef, nfill(0, Val(N)))
@generated function zerorec(::Type{T}) where {T}
    if Base._return_type(zero, (T,)) == Union{}
        exps = [:(zerorec($t)) for t in fieldtypes(T)]
        :(@_inline_meta; T($(exps...)))
    else
        :(@_inline_meta; zero(T))
    end
end
@generated function zerorec(::Type{T}) where {T <: NamedTuple}
    exps = [:(zero($t)) for t in fieldtypes(T)]
    :(@_inline_meta; T(($(exps...),)))
end
zerorec(x) = zerorec(typeof(x))
