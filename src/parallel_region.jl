# Partitioned transfers walk their color groups one group at a time: two regions
# of the same color never share a grid node, but regions of different colors do.
# Giving each group its own `Threads.@threads` costs a full fork-join per group,
# and there are `2^dim` of them per transfer. A fork-join is tens of
# microseconds, so on small problems the transfer spends more time starting and
# joining tasks than scattering particles, and adding threads makes it worse.
#
# Here the workers are spawned once per transfer and separated by a barrier that
# spins briefly and then parks, which costs a few microseconds instead. Each
# worker still takes a contiguous run of regions, so grid writes stay as local
# as they are in the sequential path, and regions keep their order within a
# group, so the results are bitwise identical to the sequential path.
#
# The trade is that the workers are held for the whole transfer, so a worker the
# OS deschedules stalls the others at the next barrier. That costs more than the
# fork-join path only when the machine is already busy with other work.

# One gate per phase, rather than one reused counter, so that a worker can park
# on the gate it is actually waiting for.
mutable struct PhaseGate
    @atomic arrived::Int
    @atomic open::Bool
    const opened::Base.Event
end
PhaseGate() = PhaseGate(0, false, Base.Event())

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
# something else. Measured on 16 cores, short phases stop getting faster around
# here.
const BARRIER_SPIN_BUDGET = 1024

@inline barrier_aborted(barrier::PhaseBarrier) = @atomic barrier.aborted

@inline function open_gate!(gate::PhaseGate)
    @atomic gate.open = true
    notify(gate.opened)
    nothing
end

# Release every worker waiting on any phase. Without this a worker that throws
# would leave the others waiting for a phase that never completes.
function abort_barrier!(barrier::PhaseBarrier)
    @atomic barrier.aborted = true
    for gate in barrier.gates
        open_gate!(gate)
    end
    nothing
end

function sync_phase!(barrier::PhaseBarrier, phase::Int)
    gate = @inbounds barrier.gates[phase]
    if (@atomic gate.arrived += 1) == barrier.nworkers
        open_gate!(gate)
        return nothing
    end
    for _ in 1:BARRIER_SPIN_BUDGET
        (@atomic gate.open) && return nothing
        ccall(:jl_cpu_pause, Cvoid, ())
        GC.safepoint()
    end
    # `Base.Event` stays set once notified, so losing the race against the last
    # worker here just means `wait` returns immediately.
    wait(gate.opened)
    nothing
end

# The number of particles a region carries, which is what `:dynamic` balances
# the runs on.
@inline region_weight(bs::BlockStrategy, region) = length(particle_indices(bs, region))
@inline region_weight(::CellStrategy, region) = 1

# `bounds[w]+1 : bounds[w+1]` is worker `w`'s run of regions.
#
# Weighting the runs by particle count only pays off when the density varies
# over the mesh in a way that follows the region order, which is exactly what a
# body occupying part of the domain looks like: a run of regions is then either
# all interior or all boundary. Measured on a sphere of particles, weighting was
# ~9% faster than equal region counts; on a uniformly filled mesh the two were
# within noise of each other.
function balanced_bounds(strat::PartitionStrategy, group, nworkers::Int, weighted::Bool)
    bounds = Vector{Int}(undef, nworkers + 1)
    nregions = length(group)
    bounds[1] = 0
    bounds[nworkers+1] = nregions

    if !weighted
        for w in 1:nworkers-1
            bounds[w+1] = (nregions * w) ÷ nworkers
        end
        return bounds
    end

    total = 0
    @inbounds for region in group
        total += region_weight(strat, region)
    end
    assigned = 0
    k = 0
    @inbounds for w in 1:nworkers-1
        target = (total * w) ÷ nworkers
        while assigned < target && k < nregions
            k += 1
            assigned += region_weight(strat, group[k])
        end
        bounds[w+1] = k
    end
    bounds
end

function run_group_chunk(work, group, bounds, w::Int)
    @inbounds for k in bounds[w]+1:bounds[w+1]
        @inline work(group[k])
    end
end

function run_group_greedy(work, group, cursor::Threads.Atomic{Int})
    nregions = length(group)
    while true
        k = Threads.atomic_add!(cursor, 1) + 1
        k > nregions && break
        @inbounds @inline work(group[k])
    end
end

# Every worker gets its own task, including the first: the loop body may leave
# things behind in task-local storage -- `@P2G_Matrix` keeps its element matrix
# there -- and running one worker in the caller would accumulate those in a
# task that outlives the transfer.
#
# A worker that throws aborts the barrier on its way out, so a transfer that
# fails partway -- an inactive `SpGrid` node, say -- releases the workers
# waiting on it instead of leaving them there. `@sync` then reports the failure
# with its backtrace intact.
function spawn_region_workers(body, nworkers::Int, nphases::Int)
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

"""
    partitioned_foreach(work, strategy, Val(scheduler))

Apply `work` to every region of every color group, running the regions of one
group in parallel and the groups one after another.

`:static` splits each group into contiguous runs of equal region count and
`:dynamic` (the default) into runs of equal particle count; `:greedy` instead
hands out single regions on demand, which balances better still but scatters
each worker's grid writes and is several times slower for it. `:nothing`, or a
single available thread, runs everything sequentially.
"""
# `work` is only handed on from here, so it needs the type parameter to be
# specialized on; the method below calls it and specializes on its own.
partitioned_foreach(work::F, strat::PartitionStrategy, ::Val{scheduler}) where {F, scheduler} =
    partitioned_foreach(work, strat, get_scheduler(Val(scheduler)))

function partitioned_foreach(work, strat::PartitionStrategy, sched::Scheduler)
    groups = threadsafe_groups(strat)

    if sched isa SequentialScheduler || Threads.nthreads() == 1
        return sequential_foreach(work, groups)
    end

    # Empty groups would only add a barrier, and every worker drops the same
    # ones, so the phases stay in step.
    active = filter(!isempty, groups)
    isempty(active) && return nothing

    # The thread count is taken as given; the only regions-based limit is not
    # spawning workers that could never take a region in any group.
    nworkers = min(Threads.nthreads(), maximum(length, active))
    nworkers == 1 && return sequential_foreach(work, active)

    if sched isa GreedyScheduler
        cursors = [Threads.Atomic{Int}(0) for _ in active]
        spawn_region_workers(nworkers, length(active) - 1) do w, barrier
            run_worker_phases(barrier, active) do k
                run_group_greedy(work, active[k], cursors[k])
            end
        end
    else
        bounds = [balanced_bounds(strat, group, nworkers, !(sched isa StaticScheduler)) for group in active]
        spawn_region_workers(nworkers, length(active) - 1) do w, barrier
            run_worker_phases(barrier, active) do k
                run_group_chunk(work, active[k], bounds[k], w)
            end
        end
    end
    nothing
end

function sequential_foreach(work, groups)
    for group in groups, region in group
        @inline work(region)
    end
    nothing
end

function run_worker_phases(phase, barrier::PhaseBarrier, active)
    for k in eachindex(active)
        @inline phase(k)
        k == lastindex(active) && break
        sync_phase!(barrier, k)
        barrier_aborted(barrier) && break
    end
    nothing
end
