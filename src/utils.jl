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

@generated function simd_otimes(x::SVec{m}, y::SVec{n}) where {m, n}
    exps = [:($(Symbol(:z_,j))[$i]) for j in 1:n for i in 1:m]
    quote
        @_inline_meta
        @nexprs $n j -> z_j = x * y[j]
        SVec(tuple($(exps...)))
    end
end
simd_otimes(x::SVec, y::SVec, others::SVec...) = simd_otimes(simd_otimes(x, y), others...)
simd_otimes(x::SVec) = x
