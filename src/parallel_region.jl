# Two regions of the same color never share a grid node; regions of different
# colors do, which is why the groups run one after another and why the result
# does not depend on which worker takes which region. Workers are spawned once
# per transfer and separated by barriers, not by a fork-join per group.

######################
# phased worker pool #
######################

# One gate per phase, so a worker parks on the gate it waits for;
# `arrived == nworkers` is what open means.
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

# Spinning pays only while the workers we wait for are running; past that,
# parking is what keeps a descheduled straggler from being starved by the very
# workers waiting on it. The bound only has to be finite, so it is a round
# number rather than a measured one.
const BARRIER_SPIN_BUDGET = 1024

@inline barrier_aborted(barrier::PhaseBarrier) = @atomic barrier.aborted

# Release every worker waiting on any phase. Without this a worker that throws
# would leave the others waiting for a phase that never completes.
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

# Every worker gets its own task, including the first: the body may leave things
# in task-local storage, as `@P2G_Matrix` does, and running one in the caller
# would strand those in a task that outlives the transfer.
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

###########################
# splitting a color group #
###########################

# `bounds[w]+1 : bounds[w+1]` is worker `w`'s run of regions.
equal_count_bounds(nregions::Int, nworkers::Int) =
    [(nregions * w) ÷ nworkers for w in 0:nworkers]

# `assign_block_ranges!` already laid this group's regions out contiguously and
# in order inside `particleindices`, so `stops` is the running particle count
# and each split point is a binary search rather than a scan.
function particle_count_bounds(bs::BlockStrategy, group, nworkers::Int)
    nregions = length(group)
    bounds = Vector{Int}(undef, nworkers + 1)
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
        # Each worker needs a region of its own and has to leave one for every
        # worker after it; a region heavier than its share would push the split
        # past the next worker's target and hand it an empty run.
        bounds[w+1] = nregions < nworkers ? k : clamp(k, w, nregions - (nworkers - w))
    end
    bounds
end

# A `CellStrategy`'s regions carry one quadrature column each, so the two splits
# coincide and the cheaper one answers for both.
weighted_bounds(strat::PartitionStrategy, group, nworkers::Int) =
    equal_count_bounds(length(group), nworkers)
weighted_bounds(bs::BlockStrategy, group, nworkers::Int) =
    particle_count_bounds(bs, group, nworkers)

group_plan(::GreedyScheduler, strat, group, nworkers::Int) = Threads.Atomic{Int}(0)
group_plan(::StaticScheduler, strat, group, nworkers::Int) = equal_count_bounds(length(group), nworkers)
group_plan(::Scheduler, strat, group, nworkers::Int) = weighted_bounds(strat, group, nworkers)

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

# The prologue runs as the transfer's first phase rather than as its own
# parallel loop, so it costs one more barrier instead of another fork-join.
run_prologue(::Nothing, nworkers::Int, w::Int) = nothing
run_prologue(prologue::P, nworkers::Int, w::Int) where {P} = (@inline prologue(nworkers, w); nothing)

"""
    partitioned_foreach(work, strategy, Val(scheduler); prologue = nothing)

Apply `work` to every region of every color group, running the regions of one
group in parallel and the groups one after another.

`:static` splits each group into contiguous runs of equal region count and
`:dynamic` (the default) into runs of equal particle count; `:greedy` instead
hands out single regions on demand, which balances better still but scatters
each worker's grid writes and is several times slower for it. `:nothing`, or a
single available thread, runs everything sequentially.

`prologue`, when given, is called as `prologue(nworkers, w)` on every worker
before the first region and is followed by a barrier, so whatever it does is
complete everywhere before any region runs. It runs on the sequential paths
too, as `prologue(1, 1)`.
"""
partitioned_foreach(work::F, strat::PartitionStrategy, ::Val{scheduler}; kwargs...) where {F, scheduler} =
    partitioned_foreach(work, strat, get_scheduler(Val(scheduler)); kwargs...)

# `work` is captured into the worker closure rather than called here, which is
# the case Julia declines to specialize on without the type parameter.
function partitioned_foreach(work::F, strat::PartitionStrategy, sched::Scheduler; prologue::P=nothing) where {F, P}
    groups = threadsafe_groups(strat)

    if sched isa SequentialScheduler || Threads.nthreads() == 1
        run_prologue(prologue, 1, 1)
        return sequential_foreach(work, groups)
    end

    # Empty groups would only add a barrier, and every worker drops the same
    # ones, so the phases stay in step.
    active = filter(!isempty, groups)
    if isempty(active)
        run_prologue(prologue, 1, 1)
        return nothing
    end

    # The thread count is taken as given; the only regions-based limit is not
    # spawning workers that could never take a region in any group.
    nworkers = min(Threads.nthreads(), maximum(length, active))
    if nworkers == 1
        run_prologue(prologue, 1, 1)
        return sequential_foreach(work, active)
    end

    ngroups = length(active)
    nphases = ngroups + (prologue === nothing ? 0 : 1)
    plans = [group_plan(sched, strat, group, nworkers) for group in active]
    spawn_region_workers(nworkers, nphases - 1) do w, barrier
        run_worker_phases(barrier, nphases) do k
            if prologue === nothing
                @inbounds run_group(work, active[k], plans[k], w)
            elseif k == 1
                run_prologue(prologue, nworkers, w)
            else
                @inbounds run_group(work, active[k-1], plans[k-1], w)
            end
        end
    end
    nothing
end
