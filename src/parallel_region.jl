# -----------------------------------------------------------------------------
#  Parallel regions
# -----------------------------------------------------------------------------

# Two regions of the same color never share a grid node; regions of different
# colors do. That is why groups run one after another, and why no result depends
# on which worker takes which region.

# ---- phased worker pool ----

mutable struct PhaseGate
    @atomic arrived::Int
    const opened::Base.Event
end
PhaseGate() = PhaseGate(0, Base.Event())

mutable struct PhaseBarrier
    const gates::Vector{PhaseGate}
    const nworkers::Int
    @atomic aborted::Bool
end
PhaseBarrier(nworkers::Integer, ngates::Integer) =
    PhaseBarrier([PhaseGate() for _ in 1:ngates], nworkers, false)

# Parking keeps a descheduled straggler from being starved by the workers
# waiting on it. The bound only has to be finite, so it is a round number.
const BARRIER_SPIN_BUDGET = 1024

@inline barrier_aborted(barrier::PhaseBarrier) = @atomic barrier.aborted

# Without this a worker that throws leaves the others waiting for a phase that
# never completes.
function abort_barrier!(barrier::PhaseBarrier)
    @atomic barrier.aborted = true
    for gate in barrier.gates
        notify(gate.opened)
    end
    nothing
end

function sync_phase!(barrier::PhaseBarrier, phase::Int)
    gate = @inbounds barrier.gates[phase]
    nworkers = barrier.nworkers
    if (@atomic gate.arrived += 1) == nworkers
        notify(gate.opened)
        return nothing
    end
    for _ in 1:BARRIER_SPIN_BUDGET
        (@atomic gate.arrived) == nworkers && return nothing
        ccall(:jl_cpu_pause, Cvoid, ())
        GC.safepoint()
    end
    # `Base.Event` stays set once notified, so losing the race against the last
    # worker here just means `wait` returns immediately.
    wait(gate.opened)
    nothing
end

# The first worker gets its own task too: the body may leave things in
# task-local storage, as `@P2G_Matrix` does, and running it in the caller would
# strand those in a task that outlives the transfer.
function spawn_region_workers(body::F, nworkers::Int, ngates::Int) where {F}
    barrier = PhaseBarrier(nworkers, ngates)
    @sync for w in 1:nworkers
        Threads.@spawn begin
            try
                body(w, barrier)
            catch
                abort_barrier!(barrier)
                rethrow()
            end
        end
    end
    nothing
end

function run_worker_phases(phase::F, barrier::PhaseBarrier, nphases::Int) where {F}
    for k in 1:nphases
        @inline phase(k)
        k == nphases && break
        sync_phase!(barrier, k)
        barrier_aborted(barrier) && break
    end
    nothing
end

# ---- splitting a color group ----

function equal_count_bounds!(bounds::Vector{Int}, nregions::Int, nworkers::Int)
    for w in 0:nworkers
        @inbounds bounds[w+1] = (nregions * w) ÷ nworkers
    end
    bounds
end
equal_count_bounds(nregions::Int, nworkers::Int) =
    equal_count_bounds!(Vector{Int}(undef, nworkers + 1), nregions, nworkers)

# `assign_block_ranges!` laid this group out contiguously and in order inside
# `particleindices`, so `stops` is already the running particle count.
function particle_count_bounds!(bounds::Vector{Int}, bs::BlockStrategy, group, nworkers::Int)
    nregions = length(group)
    bounds[1] = 0
    bounds[nworkers+1] = nregions

    blocklin = LinearIndices(nblocks(bs))
    base = @inbounds bs.starts[blocklin[first(group)]] - 1
    through(k) = @inbounds bs.stops[blocklin[group[k]]] - base
    total = through(nregions)

    # `lt` rather than `by`, because `by` would also be applied to the target,
    # which is a particle count and not a region index.
    @inbounds for w in 1:nworkers-1
        target = (total * w) ÷ nworkers
        k = searchsortedfirst(1:nregions, target; lt = (region, t) -> through(region) < t)
        # A region heavier than its share would push the split past the next
        # worker's target and hand that worker an empty run.
        bounds[w+1] = nregions < nworkers ? k : clamp(k, w, nregions - (nworkers - w))
    end
    bounds
end
particle_count_bounds(bs::BlockStrategy, group, nworkers::Int) =
    particle_count_bounds!(Vector{Int}(undef, nworkers + 1), bs, group, nworkers)

# A `CellStrategy`'s regions carry one quadrature column each, so both splits
# coincide.
weighted_bounds!(bounds::Vector{Int}, strat::PartitionStrategy, group, nworkers::Int) =
    equal_count_bounds!(bounds, length(group), nworkers)
weighted_bounds!(bounds::Vector{Int}, bs::BlockStrategy, group, nworkers::Int) =
    particle_count_bounds!(bounds, bs, group, nworkers)

group_plan!(bounds::Vector{Int}, ::StaticScheduler, strat, group, nworkers::Int) =
    equal_count_bounds!(bounds, length(group), nworkers)
group_plan!(bounds::Vector{Int}, ::Scheduler, strat, group, nworkers::Int) =
    weighted_bounds!(bounds, strat, group, nworkers)

region_scratch(strat::PartitionStrategy) = RegionScratch{eltype(threadsafe_groups(strat))}()
region_scratch(strat::Union{BlockStrategy, CellStrategy}) = strat.region_scratch

function group_plans!(scratch::RegionScratch, ::GreedyScheduler, strat, active, nworkers::Int)
    cursors = scratch.cursors
    for _ in length(cursors)+1:length(active)
        push!(cursors, Threads.Atomic{Int}(0))
    end
    for k in 1:length(active)
        @inbounds cursors[k][] = 0
    end
    cursors
end

function group_plans!(scratch::RegionScratch, sched::Scheduler, strat, active, nworkers::Int)
    bounds = scratch.bounds
    for _ in length(bounds)+1:length(active)
        push!(bounds, Int[])
    end
    for k in 1:length(active)
        @inbounds group_plan!(resize!(bounds[k], nworkers + 1), sched, strat, active[k], nworkers)
    end
    bounds
end

function run_group(work::F, group, bounds::Vector{Int}, w::Int) where {F}
    @inbounds for k in bounds[w]+1:bounds[w+1]
        @inline work(group[k])
    end
end

function run_group(work::F, group, cursor::Threads.Atomic{Int}, ::Int) where {F}
    nregions = length(group)
    while true
        k = Threads.atomic_add!(cursor, 1) + 1
        k > nregions && break
        @inbounds @inline work(group[k])
    end
end

function sequential_foreach(work::F, groups) where {F}
    for group in groups, region in group
        @inline work(region)
    end
    nothing
end

run_worker_hook(::Nothing, nworkers::Int, w::Int) = nothing
run_worker_hook(hook::H, nworkers::Int, w::Int) where {H} = (@inline hook(nworkers, w); nothing)

"""
    partitioned_foreach(work, strategy, Val(scheduler); prologue = nothing, epilogue = nothing)

Apply `work` to every region of every color group, running the regions of one
group in parallel and the groups one after another.

`:static` splits each group into contiguous runs of equal region count and
`:dynamic` (the default) into runs of equal particle count; `:greedy` instead
hands out single regions on demand, which balances better still but scatters
each worker's grid writes and is several times slower for it. `:nothing`, or a
single available thread, runs everything sequentially.

`prologue`, when given, is called as `prologue(nworkers, w)` on every worker
before the first region and is followed by a barrier, so whatever it does is
complete everywhere before any region runs. `epilogue` is the mirror: preceded
by a barrier and called after the last region, so it sees every region's writes.
Both run on the sequential paths too, as `prologue(1, 1)` / `epilogue(1, 1)`.

A worker that throws skips the epilogue, matching a separate loop after a
transfer that failed.
"""
partitioned_foreach(work::F, strat::PartitionStrategy, ::Val{scheduler}; kwargs...) where {F, scheduler} =
    partitioned_foreach(work, strat, get_scheduler(Val(scheduler)); kwargs...)

# `work` is captured into the worker closure rather than called here, which is
# the case Julia declines to specialize on without the type parameter.
function partitioned_foreach(work::F, strat::PartitionStrategy, sched::Scheduler;
                            prologue::P=nothing, epilogue::E=nothing) where {F, P, E}
    groups = threadsafe_groups(strat)

    # Both hooks still run here: a caller that put work in them relies on it
    # whichever path is taken.
    on_one_worker(groups) = (run_worker_hook(prologue, 1, 1);
                             sequential_foreach(work, groups);
                             run_worker_hook(epilogue, 1, 1))

    if sched isa SequentialScheduler || Threads.nthreads() == 1
        return on_one_worker(groups)
    end

    # Every worker drops the same empty groups, so the phases stay in step.
    scratch = region_scratch(strat)
    active = empty!(scratch.active)
    for group in groups
        isempty(group) || push!(active, group)
    end
    isempty(active) && return on_one_worker(active)

    # Regions cap the workers only when they are all there is to do: an epilogue
    # covers an index space of its own and would be left short.
    nworkers = epilogue === nothing ?
        min(Threads.nthreads(), maximum(length, active)) : Threads.nthreads()
    nworkers == 1 && return on_one_worker(active)

    ngroups = length(active)
    nphases = ngroups + (prologue === nothing ? 0 : 1) + (epilogue === nothing ? 0 : 1)
    offset = prologue === nothing ? 0 : 1
    plans = group_plans!(scratch, sched, strat, active, nworkers)
    spawn_region_workers(nworkers, nphases - 1) do w, barrier
        run_worker_phases(barrier, nphases) do k
            if prologue !== nothing && k == 1
                run_worker_hook(prologue, nworkers, w)
            elseif epilogue !== nothing && k == nphases
                run_worker_hook(epilogue, nworkers, w)
            else
                @inbounds run_group(work, active[k-offset], plans[k-offset], w)
            end
        end
    end
    nothing
end
