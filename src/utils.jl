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

function resize_fillzero!(x::AbstractVector, n::Int)
    fillzero!(resize!(x, n))
end

# merge_namedtuple_type
@pure function merge_namedtuple_type(::NamedTuple{names1, types1}, ::NamedTuple{names2, types2}) where {names1, types1, names2, types2}
    NamedTuple{(names1..., names2...), Tuple{types1.parameters..., types2.parameters...}}
end

##############
# PushVector #
##############

# https://github.com/JuliaLang/julia/issues/24909#issuecomment-419731925
mutable struct PushVector{T} <: AbstractVector{T}
    data::Vector{T}
    len::Int
end

PushVector{T}() where {T} = PushVector(Vector{T}(undef, 4), 0)

Base.size(v::PushVector) = (v.len,)
@inline function Base.getindex(v::PushVector, i)
    @boundscheck checkbounds(v, i)
    @inbounds v.data[i]
end
@inline function Base.setindex!(v::PushVector, x, i)
    @boundscheck checkbounds(v, i)
    @inbounds v.data[i] = x
    v
end

function Base.push!(v::PushVector, i)
    v.len += 1
    if v.len > length(v.data)
        resize!(v.data, v.len * 2)
    end
    v.data[v.len] = i
    v
end

Base.empty!(v::PushVector) = (v.len=0; v)

finish!(v::PushVector) = resize!(v.data, v.len)

###########
# AllTrue #
###########

struct AllTrue end
@pure Base.getindex(::AllTrue, i...) = true
@pure Base.all(::AllTrue) = true
@pure Base.view(::AllTrue, i...) = AllTrue()
