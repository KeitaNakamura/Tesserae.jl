const DEBUG = Preferences.@load_preference("debug_mode", false)

@static if DEBUG
    @eval macro debug(ex)
        return :($(esc(ex)))
    end
else
    @eval macro debug(ex)
         return nothing
    end
end

#############
# Utilities #
#############

nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

# zero_recursive
@generated function zero_recursive(::Type{T}) where {T}
    isbitstype(T) || return :(throw(ArgumentError("`zero_recursive` supports only `isbitstype`, got $T")))
    :(@_inline_meta; zero(T))
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

# commas
commas(num::Integer) = replace(string(num), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")

# getx
getx(x) = getproperty(x, first(propertynames(x)))
getx(x::Vec) = x
getx(x::Vector{<: Vec}) = x

# lazy_getindex
@inline function lazy_getindex(x::StructArray, i...)
    @boundscheck checkbounds(x, i...)
    @inbounds LazyRow(x, i...)
end
@inline function lazy_getindex(x::AbstractArray, i...)
    @boundscheck checkbounds(x, i...)
    @inbounds x[i...]
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

##################
# threaded macro #
##################

const THREADED = Preferences.@load_preference("threaded_macro", true)

macro threaded(schedule::QuoteNode, expr)
    @assert Meta.isexpr(expr, :for)
    quote
        if Threads.nthreads() > 1 && $THREADED
            Threads.@threads $schedule $expr
        else
            $expr
        end
    end |> esc
end
macro threaded(expr)
    quote
        @threaded :dynamic $expr
    end |> esc
end

const SHOWPROGRESS = Preferences.@load_preference("showprogress_macro", true)

"""
```
@showprogress while t < t_stop
    # computation...
end
```

displays progress of `while` loop.
"""
macro showprogress(expr)
    SHOWPROGRESS || return esc(expr)
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
        speed = t_current - prog.tlast
        ProgressMeter.update!(prog,
                              min(floor(Int, ($t/$t_stop)*$thresh), $thresh);
                              showvalues = [(:Elapsed, ProgressMeter.durationstring(elapsed)),
                                            (:Iterations, commas(count)),
                                            (:Speed, lstrip(ProgressMeter.speedstring(speed)))])
    end
    push!(blk.args, inner)
    quote
        prog = ProgressMeter.Progress($thresh)
        count = 0
        $expr
        ProgressMeter.finish!(prog)
    end
end
