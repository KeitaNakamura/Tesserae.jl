# -----------------------------------------------------------------------------
#  Threading primitives
# -----------------------------------------------------------------------------

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

# Chunks past the last piece come out empty.
@inline chunk_range(chunk_id::Int, chunksize::Int, n::Int) =
    ((chunk_id - 1) * chunksize + 1) : min(chunk_id * chunksize, n)

# `f` is only handed on to `_tforeach`, which is the one case where Julia skips
# specializing on a function argument, so it takes the type parameter here. The
# `_tforeach` methods below call `f` and specialize without one.
function tforeach(f::F, iter, scheduler=DynamicScheduler(); kwargs...) where {F}
    if Threads.nthreads() > 1
        _tforeach(f, iter, get_scheduler(scheduler); kwargs...)
    else
        _tforeach(f, iter, SequentialScheduler(); kwargs...)
    end
end

# The single point to change for a custom threading backend.
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

What the three parallel schedulers select depends on what is being threaded. On
a plain loop, and on [`@G2P`](@ref), they pick the corresponding
`Threads.@threads` variant. On a partitioned transfer -- [`@P2G`](@ref),
[`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) given a [`ThreadPartition`](@ref) --
they instead pick how each color group is divided among the workers: `:static`
by region count, `:dynamic` by particle count, and `:greedy` one region at a
time on demand, which balances best and is several times slower because it
scatters each worker's grid writes.

See also [`ThreadPartition`](@ref).

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
           (expr.args[1] in (Symbol("@P2G"), Symbol("@G2P"), Symbol("@G2P2G"), Symbol("@P2G_Matrix"), Symbol("@foreach")))
        insert!(expr.args, 3, schedule)
        esc(expr)
    else
        error("wrong usage for @threaded")
    end
end
