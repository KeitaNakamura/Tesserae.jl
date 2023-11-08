nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

isapproxzero(x::Number) = abs(x) < sqrt(eps(typeof(x)))

# zero_recursive
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
    @assert isbitstype(eltype(x))
    fill!(x, zero_recursive(eltype(x)))
    x
end

# rename property names
@generated function rename(::Val{srcnames}, ::Val{before}, ::Val{after}) where {srcnames, before, after}
    @assert length(before) == length(after)
    before_new = collect(before)
    after_new = collect(after)
    for (bf, af) in zip(before, after)
        if bf in srcnames && af in srcnames && !(af in before)
            push!(before_new, af)
            push!(after_new, Symbol(:__, af, :__))
        end
    end
    nt = (; zip(before_new, after_new)...)
    map(name -> get(nt, name, name), srcnames)
end
function rename(A::NamedTuple, before::Val, after::Val)
    newnames = rename(Val(keys(A)), before, after)
    NamedTuple{newnames}(values(A))
end
function rename(A::StructArray, before::Val, after::Val)
    StructArray(rename(StructArrays.components(A), before, after))
end

macro rename(src, list...)
    for ex in list
        @assert Meta.isexpr(ex, :call) && ex.args[1] == :(=>)
    end
    before = Val(tuple(Symbol[ex.args[2] for ex in list]...))
    after  = Val(tuple(Symbol[ex.args[3] for ex in list]...))
    esc(:(Marble.rename($src, $before, $after)))
end

# combine properties
@generated function combine_names(::Val{dest}, ::Val{src}) where {dest, src}
    src_tmp = map(x->Symbol(:____, x, :____), src)
    combined = (dest..., src_tmp...)
    rename(Val(combined), Val(src_tmp), Val(src))
end

function combine(dest::NamedTuple, src::NamedTuple)
    newnames = combine_names(Val(keys(dest)), Val(keys(src)))
    NamedTuple{newnames}((values(dest)..., values(src)...))
end
function combine(A::StructArray, src::NamedTuple)
    StructArray(combine(StructArrays.components(A), src))
end
function combine(A::StructArray, B::StructArray)
    combine(A, StructArrays.components(B))
end

@inline flatarray(A::AbstractArray{Vec{dim, T}, dim}) where {dim, T} = reinterpret(reshape, T, A)
@inline function flatview(A::AbstractArray{<: Vec{dim}, dim}, flatfreeinds::AbstractVector{CartesianIndex{N}}) where {dim, N}
    @assert dim+1 == N
    @boundscheck checkbounds(flatarray(A), flatfreeinds)
    @inbounds view(flatarray(A), flatfreeinds)
end

# commas
commas(num::Integer) = replace(string(num), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")

#####################
# NormalVectorArray #
#####################

struct NormalVectorArray{dim} <: AbstractArray{Vec{dim, Int}, dim}
    size::Dims{dim}
end
Base.size(A::NormalVectorArray) = A.size
@inline function Base.getindex(A::NormalVectorArray{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    Vec{dim, Int}() do i
        firstindex(A, i) == I[i] && return -1
        lastindex(A, i) == I[i] && return 1
        0
    end
end

############
# MapArray #
############

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
Base.IndexStyle(::Type{<: MapArray{<: Any, <: Any, F, Args}}) where {F, Args} = IndexStyle(Base._return_type(map, Tuple{F, Args.parameters...}))
@inline function Base.getindex(A::MapArray, i::Integer...)
    @boundscheck checkbounds(A, i...)
    @inbounds A.f(getindex.(A.args, i...)...)
end

#############
# FillArray #
#############

struct FillArray{T, N} <: AbstractArray{T, N}
    value::T
    dims::Dims{N}
end
FillArray(value::Any, dims::Int...) = FillArray(value, dims)
Base.size(A::FillArray) = A.dims
Base.IndexStyle(::Type{<: FillArray}) = IndexLinear()
@inline function Base.getindex(A::FillArray, i::Integer)
    @boundscheck checkbounds(A, i)
    A.value
end

####################
# CoordinateSystem #
####################

abstract type CoordinateSystem end
struct DefaultSystem <: CoordinateSystem end
struct PlaneStrain   <: CoordinateSystem end
struct Axisymmetric  <: CoordinateSystem end

#########
# Trues #
#########

struct Trues{N} <: AbstractArray{Bool, N}
    dims::Dims{N}
end
Base.size(t::Trues) = t.dims
Base.IndexStyle(::Type{<: Trues}) = IndexLinear()
@inline function Base.getindex(t::Trues, i::Integer)
    @boundscheck checkbounds(t, i)
    true
end

####################
# parallel_foreach #
####################

function parallel_foreach(f, iter, schedule::Symbol; ntasks::Integer)
    if ntasks == 1
        foreach(f, iter)
    else
        if schedule == :static
            Threads.@threads :static for x in iter
                f(x)
            end
        elseif schedule == :dynamic
            chunks = Iterators.partition(iter, max(1, length(iter) ÷ ntasks))
            tasks = map(chunks) do chunk
                Threads.@spawn foreach(f, chunk)
            end
            fetch.(tasks)
        else
            throw(ArgumentError("unsupported schedule argument in Marble.parallel_foreach"))
        end
    end
    nothing
end

########
# SIMD #
########

@inline SIMD.Vec(x::Vec) = SVec(Tuple(x))
@inline SIMD.Vec{dim,T}(x::Vec{dim,T}) where {dim,T<:SIMDTypes} = SVec(Tuple(x))
@inline SIMD.Vec{dim,T}(x::Vec{dim,U}) where {dim,T<:SIMDTypes,U<:SIMDTypes} = SVec(convert(Vec{dim,T}, x))

#################
# @showprogress #
#################

"""
```
@showprogress while t < t_stop
    # computation goes here
end
```

displays progress of `while` loop.
"""
macro showprogress(expr)
    @assert Meta.isexpr(expr, :while)
    cnd, blk = expr.args[1], expr.args[2]
    @assert Meta.isexpr(cnd, :call) && cnd.args[1] == :<
    thresh = 10000
    t, t_stop = esc(cnd.args[2]), esc(cnd.args[3])
    map!(esc, cnd.args, cnd.args)
    map!(esc, blk.args, blk.args)
    inner = quote
        count += 1
        t_current = time()
        elapsed = t_current - prog.tinit
        speed = lstrip(ProgressMeter.speedstring(t_current-prog.tlast))
        speed_ave = lstrip(ProgressMeter.speedstring(elapsed/count))
        ProgressMeter.update!(prog,
                              min(floor(Int, ($t/$t_stop)*$thresh), $thresh);
                              showvalues = [(:Elapsed, ProgressMeter.durationstring(elapsed)),
                                            (:Iterations, commas(count)),
                                            (:Speed, string(speed, " (Ave. ", speed_ave, ")"))])
    end
    push!(blk.args, inner)
    quote
        prog = ProgressMeter.Progress($thresh)
        count = 0
        $expr
        ProgressMeter.finish!(prog)
    end
end
