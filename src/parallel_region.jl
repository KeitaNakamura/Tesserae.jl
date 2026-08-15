# Partitioned transfers walk their color groups one group at a time: two regions
# of the same color never share a grid node, but regions of different colors do.
# Giving each group its own `Threads.@threads` costs a full fork-join per group,
# and there are `2^dim` of them per transfer. A fork-join is tens of
# microseconds, so on small problems the transfer spends more time starting and
# joining tasks than scattering particles, and adding threads makes it worse.
#
# Here the workers are spawned once per transfer and separated by a barrier that
# spins briefly and then parks, which costs a few microseconds instead. Results
# stay bitwise identical to the sequential path because same-color regions share
# no node, so no sum is ever reassociated no matter which worker takes which
# region or in what order. Under `:static` and `:dynamic` each worker also takes
# a contiguous run, which keeps grid writes as local as the sequential path;
# `:greedy` gives that up for load balancing, not for reproducibility.
#
# The trade is that the workers are held for the whole transfer, so a worker the
# OS deschedules stalls the others at the next barrier. That costs more than the
# fork-join path only when the machine is already busy with other work.

######################
# phased worker pool #
######################

# Nothing below this line knows about partitions: it runs `nphases` phases on a
# pool of workers with a barrier between them. `partitioned_foreach` is the
# adapter that turns color groups into phases.

# One gate per phase, so a worker parks on the gate it is actually waiting for.
# `arrived == nworkers` is what "open" means; there is no separate flag to keep
# in step with it.
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
PhaseBarrier(nworkers::Integer, nphases::Integer) =
    PhaseBarrier([PhaseGate() for _ in 1:nphases], nworkers, false)

# Spinning pays off only while the workers we wait for are running. Past that,
# parking is what keeps a straggler from being starved by the very workers
# waiting on it -- which is what happens when the machine is also busy with
# something else.
#
# What this bound has to be is finite. It is not picked to optimise anything:
# any value that ends the spin before a descheduled worker can be starved does
# the job, which is why it is a round number and not a measured one.
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

# Every worker gets its own task, including the first: the loop body may leave
# things behind in task-local storage -- `@P2G_Matrix` keeps its element matrix
# there -- and running one worker in the caller would accumulate those in a
# task that outlives the transfer.
#
# A worker that throws aborts the barrier on its way out, so a transfer that
# fails partway releases the workers waiting on it instead of leaving them
# parked forever. `@sync` then reports the failure with its backtrace intact.
function spawn_region_workers(body::F, nworkers::Int, nphases::Int) where {F}
    barrier = PhaseBarrier(nworkers, nphases)
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

##########################
# splitting a color group #
##########################

# `bounds[w]+1 : bounds[w+1]` is worker `w`'s run of regions. Runs are
# contiguous either way, so a worker's grid writes stay together.
equal_count_bounds(nregions::Int, nworkers::Int) =
    [(nregions * w) ÷ nworkers for w in 0:nworkers]

# Splitting by particle count instead of region count pays off when the density
# varies over the mesh in a way that follows the region order, which is what a
# body occupying part of the domain looks like: a run of regions is then either
# all interior or all boundary. On a uniformly filled mesh the two are the same.
#
# `assign_block_ranges!` already laid this group's regions out contiguously and
# in order inside `particleindices`, so `stops` *is* the running particle count
# and each split point is a binary search rather than a scan over every region.
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
        # Every worker needs a region of its own, and has to leave one for each
        # worker after it. A region heavier than its share pushes the split past
        # the next worker's target, which without this would hand that worker an
        # empty run -- and an idle worker costs a whole phase, so it hurts most
        # when the regions barely outnumber the workers.
        bounds[w+1] = nregions < nworkers ? k : clamp(k, w, nregions - (nworkers - w))
    end
    bounds
end

# Only a `BlockStrategy` can say how many particles a region carries. A
# `CellStrategy`'s regions carry one quadrature column each, so the two splits
# coincide and the cheaper one answers for both.
weighted_bounds(strat::PartitionStrategy, group, nworkers::Int) =
    equal_count_bounds(length(group), nworkers)
weighted_bounds(bs::BlockStrategy, group, nworkers::Int) =
    particle_count_bounds(bs, group, nworkers)

# `:static` asks for equal region counts, `:dynamic` for equal particle counts,
# and `:greedy` hands out single regions through a shared cursor.
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

# Work that has to finish before any region runs -- `@P2G` zeroing the grid
# fields it assigns is the one caller -- goes in as the region's first phase
# rather than as its own parallel loop before it.
#
# Its own loop would cost a whole fork-join, which is priced by the thread count
# rather than by the work and climbs steeply once `-t` passes `Sys.CPU_THREADS`.
# That is more than the zeroing itself at most grid sizes, and it grows just as
# the machine gets wider.
#
# The workers below are already spawned and already separated by barriers, so a
# prologue costs one more barrier instead. A barrier over workers that are
# already running and spinning cannot cost more than starting and joining them,
# which is the whole argument -- it does not rest on a number.
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
