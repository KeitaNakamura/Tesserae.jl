struct Index{dim}
    i::Int
    I::CartesianIndex{dim}
end
@inline Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::Index) = checkindex(Bool, inds, i.i)
@inline _to_indices(::IndexLinear, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (I[1].i, Base.tail(I)...))
@inline _to_indices(::IndexCartesian, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (Tuple(I[1].I)..., Base.tail(I)...))
@inline Base.to_indices(A, inds, I::Tuple{Index, Vararg{Any}}) = _to_indices(IndexStyle(A), A, inds, I)

nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

promote_tuple_length() = -1
promote_tuple_length(xs::Type{<: NTuple{N, Any}}...) where {N} = N
@generated function broadcast_tuple(f, xs::Vararg{Any, N}) where {N}
    L = promote_tuple_length([x for x in xs if x <: Tuple]...)
    if L == -1 # no tuples
        quote
            @_inline_meta
            @_propagate_inbounds_meta
            f(xs...)
        end
    else
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
end

isapproxzero(x::Number) = abs(x) < sqrt(eps(typeof(x)))

macro _inline_propagate_inbounds_meta()
    quote
        Base.@_inline_meta
        Base.@_propagate_inbounds_meta
    end
end

# elzero
elzero(x) = zero(eltype(x))

# recursive_zero
recursive_zero(::Type{Array{T, N}}) where {T, N} = Array{T, N}(undef, nfill(0, Val(N)))
@generated function recursive_zero(::Type{T}) where {T}
    if Base._return_type(zero, Tuple{T}) == Union{}
        exps = [:(recursive_zero($t)) for t in fieldtypes(T)]
        :(@_inline_meta; T($(exps...)))
    else
        :(@_inline_meta; zero(T))
    end
end
@generated function recursive_zero(::Type{T}) where {T <: NamedTuple}
    exps = [:(recursive_zero($t)) for t in fieldtypes(T)]
    :(@_inline_meta; T(($(exps...),)))
end
recursive_zero(x) = recursive_zero(typeof(x))
