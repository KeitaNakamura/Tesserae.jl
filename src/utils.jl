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

################
# Order/Degree #
################

struct Order{n}
    Order{n}() where {n} = new{n::Int}()
end
Order(n::Int) = Order{n}()

struct Degree{n}
    Degree{n}() where {n} = new{n::Int}()
end
Degree(n::Int) = Degree{n}()
const Linear    = Degree{1}
const Quadratic = Degree{2}
const Cubic     = Degree{3}
const Quartic   = Degree{4}
const Quintic   = Degree{5}

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
    fill!(x, zero_recursive(eltype(x)))
    x
end
function fillzero!(x::StructArray)
    StructArrays.foreachfield(fillzero!, x)
    x
end

# fastsum
@inline function fastsum(f, iter)
    ret = zero(Base._return_type(f, Tuple{eltype(iter)}))
    @simd for x in iter
        ret += @inline f(x)
    end
    ret
end

# commas
commas(num::Integer) = replace(string(num), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")

# getx
getx(x) = getproperty(x, first(propertynames(x)))
getx(x::Vec) = x
getx(x::Vector{<: Vec}) = x

# flatten_tuple
@generated function flatten_tuple(x::Tuple{Vararg{Tuple, N}}) where {N}
    exps = [Expr(:..., :(x[$i])) for i in 1:N]
    :(tuple($(exps...)))
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
Base.IndexStyle(::Type{<: MapArray}) = IndexCartesian()
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

abstract type Scheduler end
struct StaticScheduler     <: Scheduler end
struct DynamicScheduler    <: Scheduler end
struct GreedyScheduler     <: Scheduler end
struct SequentialScheduler <: Scheduler end

get_scheduler(sch::Scheduler) = sch
get_scheduler(sch::Symbol) = get_scheduler(Val(sch))
get_scheduler(::Val{:static})  = StaticScheduler()
get_scheduler(::Val{:dynamic}) = DynamicScheduler()
get_scheduler(::Val{:greedy})  = GreedyScheduler()
get_scheduler(::Val{:nothing}) = SequentialScheduler()

function tforeach(f, iter, scheduler=DynamicScheduler(); kwargs...)
    if Threads.nthreads() > 1
        _tforeach(f, iter, get_scheduler(scheduler); kwargs...)
    else
        _tforeach(f, iter, SequentialScheduler(); kwargs...)
    end
end

# Modify the following funcitons for custom multi-threading. For now, just use Threads.@threads.
function _tforeach(f, iter, ::StaticScheduler)
    Threads.@threads :static for i in iter
        @inline f(i)
    end
end
function _tforeach(f, iter, ::DynamicScheduler)
    Threads.@threads :dynamic for i in iter
        @inline f(i)
    end
end
function _tforeach(f, iter, ::GreedyScheduler)
    Threads.@threads :greedy for i in iter
        @inline f(i)
    end
end
function _tforeach(f, iter, ::SequentialScheduler)
    for i in iter
        @inline f(i)
    end
end

"""
    @threaded [scheduler] for ...
    @threaded [scheduler] @P2G ...
    @threaded @P2G ...

A macro similar to `Threads.@threads`, but also works with
[`@P2G`](@ref), [`@G2P`](@ref), [`@G2P2G`](@ref), and [`@P2G_Matrix`](@ref) macros for particle-grid transfers.

The optional `scheduler` can be `:static`, `:dynamic`, `:greedy`, or `:nothing`
(sequential execution). The default is `:dynamic`.

See also [`ColorPartition`](@ref).

!!! note
    If multi-threading is disabled or only one thread is available,
    this macro falls back to sequential execution.

# Examples
```julia
# Parallel loop
@threaded for i in 1:100
    println(i)
end

# Grid-to-particle transfer
@threaded @G2P grid=>i particles=>p weights=>ip begin
    v[p] = @∑ w[ip] * v[i]
end
```
"""
macro threaded(expr)
    threaded_expr(QuoteNode(:dynamic), expr)
end

macro threaded(schedule::QuoteNode, expr)
    threaded_expr(schedule, expr)
end

function threaded_expr(schedule::QuoteNode, expr::Expr)
    if Meta.isexpr(expr, :for)
        head = expr.args[1]
        index = esc(head.args[1])
        iter = esc(head.args[2])
        body = esc(expr.args[2])
        quote
            Tesserae.tforeach($iter, $schedule) do $index
                $body
            end
        end
    elseif Meta.isexpr(expr, :macrocall) &&
           (expr.args[1] in (Symbol("@P2G"), Symbol("@G2P"), Symbol("@G2P2G"), Symbol("@P2G_Matrix")))
        insert!(expr.args, 3, schedule)
        esc(expr)
    else
        error("wrong usage for @threaded")
    end
end

const SHOWPROGRESS = Preferences.@load_preference("enable_showprogress_macro", true)

#########
# tmul! #
#########

# multithreading C = Aᵀ B α + C β
function tmul!(C::StridedVecOrMat{T}, A::SparseMatrixCSC{T}, B::StridedVecOrMat{T}, α, β) where {T <: Real}
    rows = rowvals(A)
    vals = nonzeros(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k in 1:size(C, 2)
        @threaded for j in 1:size(A, 2)
            @inbounds begin
                tmp = zero(T)
                for i in nzrange(A, j)
                    row = rows[i]
                    val = vals[i]
                    tmp += val * B[row, k]
                end
                C[j, k] += tmp * α
            end
        end
    end
    C
end
tmul!(C::StridedVecOrMat, A::SparseMatrixCSC, B::StridedVecOrMat) = tmul!(C, A, B, true, false)

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
        speed = t_current - t_last
        ProgressMeter.update!(prog,
                              min(floor(Int, ($t/$t_stop)*$thresh), $thresh);
                              showvalues = [(:Elapsed, ProgressMeter.durationstring(elapsed)),
                                            (:Iterations, commas(count)),
                                            (:Speed, lstrip(ProgressMeter.speedstring(speed)))])
        t_last = time()
    end
    push!(blk.args, inner)
    quote
        prog = ProgressMeter.Progress($thresh)
        count = 0
        t_last = time()
        $expr
        ProgressMeter.finish!(prog)
    end
end
