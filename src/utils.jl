nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

promote_tuple_length() = -1
promote_tuple_length(xs::Type{<: NTuple{N, Any}}...) where {N} = N
# apply map calculations only for tuples
# if one of the argunents is not tuple, treat it as scalar
@generated function map_tuple(f, xs::Vararg{Any, N}) where {N}
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

# zero_recursive
zero_recursive(::Type{Array{T, N}}) where {T, N} = Array{T, N}(undef, nfill(0, Val(N)))
@generated function zero_recursive(::Type{T}) where {T}
    if Base._return_type(zero, Tuple{T}) == Union{}
        exps = [:(zero_recursive($t)) for t in fieldtypes(T)]
        :(@_inline_meta; T($(exps...)))
    else
        :(@_inline_meta; zero(T))
    end
end
@generated function zero_recursive(::Type{T}) where {T <: Union{Tuple, NamedTuple}}
    exps = [:(zero_recursive($t)) for t in fieldtypes(T)]
    :(@_inline_meta; T(($(exps...),)))
end
zero_recursive(x) = zero_recursive(typeof(x))

# fillzero!
function fillzero!(x::AbstractArray)
    @simd for i in eachindex(x)
        @inbounds x[i] = zero_recursive(eltype(x))
    end
    x
end
