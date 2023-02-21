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

# rename property names
function rename(A::NamedTuple{srcnames}, ::Val{before}, ::Val{after}) where {srcnames, before, after}
    @assert length(before) == length(after)
    nt = (; zip(before, after)...)
    newnames = map(name -> get(nt, name, name), srcnames)
    NamedTuple{newnames}(values(A))
end
function rename(A::StructArray, b::Val, a::Val)
    StructArray(rename(StructArrays.components(A), b, a))
end

macro rename(src, list...)
    for ex in list
        @assert Meta.isexpr(ex, :call) && ex.args[1] == :(=>)
    end
    before = Val(tuple(Symbol[ex.args[2] for ex in list]...))
    after  = Val(tuple(Symbol[ex.args[3] for ex in list]...))
    esc(:(Marble.rename($src, $before, $after)))
end

####################
# CoordinateSystem #
####################

abstract type CoordinateSystem end
struct NormalSystem <: CoordinateSystem end
struct PlaneStrain  <: CoordinateSystem end
struct Axisymmetric <: CoordinateSystem end

#########
# Trues #
#########

struct Trues{N} <: AbstractArray{Bool, N}
    dims::Dims{N}
end
Base.size(t::Trues) = t.dims
Base.IndexStyle(::Type{<: Trues}) = IndexLinear()
@inline function Base.getindex(t::Trues, i::Int)
    @boundscheck checkbounds(t, i)
    true
end

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

########
# SIMD #
########

@inline SIMD.Vec(x::Vec) = SVec(Tuple(x))
@inline StaticArrays.SVector(x::SVec) = SVector(Tuple(x))

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
    n = v.len += 1
    n > length(v.data) && resize!(v.data, 2n)
    @inbounds v.data[n] = i
    v
end

Base.empty!(v::PushVector) = (v.len=0; v)
finish!(v::PushVector) = resize!(v.data, v.len)
