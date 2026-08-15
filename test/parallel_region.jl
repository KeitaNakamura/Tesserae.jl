@testset "partitioned_foreach" begin
    mesh = CartesianMesh(0.05, (0,1), (0,1))
    particles = generate_particles(@NamedTuple{x::Vec{2, Float64}}, mesh)
    bs = Tesserae.BlockStrategy(mesh)
    update!(bs, particles.x)

    groups = Tesserae.threadsafe_groups(bs)
    regions = reduce(vcat, groups)
    @test !isempty(regions)

    @testset "visits every region exactly once ($schedule)" for schedule in (:nothing, :static, :dynamic, :greedy)
        visits = Dict(region => Threads.Atomic{Int}(0) for region in regions)
        Tesserae.partitioned_foreach(bs, Val(schedule)) do region
            Threads.atomic_add!(visits[region], 1)
        end
        @test all(v -> v[] == 1, values(visits))
    end

    # `@P2G` zeroes its grid fields in the prologue, so a path that skips it
    # scatters into a grid still holding the previous step's values -- silently,
    # and only on whichever path was missed. Every early return has to run it,
    # the sequential ones included, since those are what a user hits with no
    # `@threaded` at all.
    @testset "the prologue runs on every path ($schedule)" for schedule in (:nothing, :static, :dynamic, :greedy)
        empty_bs = Tesserae.BlockStrategy(mesh)
        update!(empty_bs, Vec{2,Float64}[])
        @test all(isempty, Tesserae.threadsafe_groups(empty_bs))

        for (label, strat) in (("with regions", bs), ("no regions at all", empty_bs))
            ran = Threads.Atomic{Int}(0)
            covered = Threads.Atomic{Int}(0)
            Tesserae.partitioned_foreach(strat, Val(schedule);
                                         prologue = (nchunks, chunk_id) -> begin
                                             Threads.atomic_add!(ran, 1)
                                             chunk_id == 1 && Threads.atomic_add!(covered, nchunks)
                                         end) do region
                nothing
            end
            # One call per worker, and the workers between them see every chunk.
            @test ran[] ≥ 1
            @test ran[] == covered[]
        end
    end

    # A prologue that throws has to fail the same way a region does, rather than
    # leaving the workers parked on a barrier that never opens.
    @testset "a failing prologue throws instead of hanging ($schedule)" for schedule in (:static, :dynamic, :greedy)
        @test_throws "prologue failed" Tesserae.partitioned_foreach(bs, Val(schedule);
                                                                    prologue = (_, _) -> error("prologue failed")) do region
            nothing
        end
    end

    # A worker that throws must release the workers already waiting on the phase
    # barrier. If it does not, this hangs rather than fails.
    @testset "one failing region throws instead of hanging ($schedule)" for schedule in (:static, :dynamic, :greedy)
        # Fail inside the first group, so the remaining groups still have
        # barriers that the surviving workers would otherwise wait on forever.
        victim = last(first(groups))
        @test_throws "region failed" Tesserae.partitioned_foreach(bs, Val(schedule)) do region
            region == victim && error("region failed")
        end
    end

    # The loop body may stash things in task-local storage -- @P2G_Matrix keeps
    # its element matrix there -- so running a worker in the calling task would
    # leave them in a task that outlives the transfer.
    @testset "work never runs on the calling task ($schedule)" for schedule in (:static, :dynamic, :greedy)
        if Threads.nthreads() > 1
            caller = current_task()
            off_caller = Threads.Atomic{Bool}(true)
            Tesserae.partitioned_foreach(bs, Val(schedule)) do region
                current_task() === caller && (off_caller[] = false)
            end
            @test off_caller[]
        end
    end

    # A region heavier than its share pushes the weighted split past the next
    # worker's target, which used to leave that worker with nothing to do. With
    # one region per worker that idles a whole phase, which is where it showed.
    @testset "no worker is left with an empty run" begin
        for nworkers in 1:8
            for group in Tesserae.threadsafe_groups(bs)
                isempty(group) && continue
                nregions = length(group)
                nregions < nworkers && continue
                for bounds in (Tesserae.equal_count_bounds(nregions, nworkers),
                               Tesserae.particle_count_bounds(bs, group, nworkers))
                    @test bounds[1] == 0 && bounds[end] == nregions
                    @test issorted(bounds)
                    @test all(w -> bounds[w+1] > bounds[w], 1:nworkers)
                    covered = reduce(vcat, [group[bounds[w]+1:bounds[w+1]] for w in 1:nworkers])
                    @test covered == group
                end
            end
        end
    end

    @testset "an unknown scheduler is rejected" begin
        @test_throws MethodError Tesserae.partitioned_foreach(identity, bs, Val(:sequential))
    end

    @testset "no work is done after a failure ($schedule)" for schedule in (:static, :dynamic, :greedy)
        victim = last(first(groups))
        after = Threads.Atomic{Int}(0)
        try
            Tesserae.partitioned_foreach(bs, Val(schedule)) do region
                region == victim && error("region failed")
                # Regions of later groups must not run once the transfer failed.
                !(region in first(groups)) && Threads.atomic_add!(after, 1)
            end
        catch
        end
        @test after[] == 0
    end
end

@testset "PhaseBarrier" begin
    @testset "releases all workers each phase" begin
        nworkers = 4
        nphases = 8
        barrier = Tesserae.PhaseBarrier(nworkers, nphases - 1)
        counts = [Threads.Atomic{Int}(0) for _ in 1:nphases]
        @sync for _ in 1:nworkers
            Threads.@spawn for phase in 1:nphases
                Threads.atomic_add!(counts[phase], 1)
                phase == nphases || Tesserae.sync_phase!(barrier, phase)
            end
        end
        @test all(c -> c[] == nworkers, counts)
    end

    @testset "abort releases waiting workers" begin
        barrier = Tesserae.PhaseBarrier(2, 1)
        waiter = Threads.@spawn Tesserae.sync_phase!(barrier, 1)
        Tesserae.abort_barrier!(barrier)
        @test timedwait(() -> istaskdone(waiter), 10.0) === :ok
        @test Tesserae.barrier_aborted(barrier)
    end
end
