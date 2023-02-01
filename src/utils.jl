nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

isapproxzero(x::Number) = abs(x) < sqrt(eps(typeof(x)))

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

# merge_namedtuple_type
@pure function merge_namedtuple_type(::NamedTuple{names1, types1}, ::NamedTuple{names2, types2}) where {names1, types1, names2, types2}
    NamedTuple{(names1..., names2...), Tuple{types1.parameters..., types2.parameters...}}
end

###########
# AllTrue #
###########

struct AllTrue end
@pure Base.getindex(::AllTrue, i...) = true

#############
# @threaded #
#############

macro threaded(expr)
    @assert Meta.isexpr(expr, :for)
    # insert @inbounds macro
    expr.args[2] = quote
        @inbounds begin
            $(expr.args[2])
        end
    end
    quote
        let
            if Threads.nthreads() == 1
                $(expr)
            else
                Threads.@threads $(expr)
            end
        end
    end |> esc
end
