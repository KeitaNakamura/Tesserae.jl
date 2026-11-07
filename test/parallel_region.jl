@testset "partitioned_foreach" begin
    mesh = CartesianMesh(0.05, (0,1), (0,1))
    particles = generate_particles(@NamedTuple{x::Vec{2, Float64}}, mesh)
    bs = Tesserae.CPUBlockStrategy(mesh)
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
        empty_bs = Tesserae.CPUBlockStrategy(mesh)
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

    # The epilogue carries `@P2G`'s grid-node half, which reads what the regions
    # wrote. Missing it on one path leaves those equations unevaluated; running
    # it early reads half-scattered values. Both are silent.
    @testset "the epilogue runs on every path, after every region ($schedule)" for schedule in (:nothing, :static, :dynamic, :greedy)
        empty_bs = Tesserae.CPUBlockStrategy(mesh)
        update!(empty_bs, Vec{2,Float64}[])

        for (label, strat) in (("with regions", bs), ("no regions at all", empty_bs))
            ran = Threads.Atomic{Int}(0)
            covered = Threads.Atomic{Int}(0)
            regions = Threads.Atomic{Int}(0)
            seen = Threads.Atomic{Int}(0)
            Tesserae.partitioned_foreach(strat, Val(schedule);
                                         epilogue = (nchunks, chunk_id) -> begin
                                             Threads.atomic_add!(ran, 1)
                                             chunk_id == 1 && Threads.atomic_add!(covered, nchunks)
                                             Threads.atomic_add!(seen, regions[])
                                         end) do region
                Threads.atomic_add!(regions, 1)
            end
            nregions = sum(length, Tesserae.threadsafe_groups(strat))
            @test ran[] ≥ 1
            @test ran[] == covered[]
            @test regions[] == nregions
            # Every worker saw the full region count, so the barrier held.
            @test seen[] == ran[] * nregions
        end
    end

    # A prologue that throws has to fail the same way a region does, rather than
    # leaving the workers parked on a barrier that never opens.
    @testset "a failing prologue throws instead of hanging ($schedule)" for schedule in (:static, :dynamic, :greedy)
        after = Threads.Atomic{Int}(0)
        @test_throws "prologue failed" Tesserae.partitioned_foreach(bs, Val(schedule);
                                                                    prologue = (_, w) -> w == 1 && error("prologue failed"),
                                                                    epilogue = (_, _) -> Threads.atomic_add!(after, 1)) do region
            Threads.atomic_add!(after, 1)
        end
        @test after[] == 0
    end

    # A worker that throws must release the workers already waiting on the phase
    # barrier. If it does not, this hangs rather than fails.
    @testset "one failing region throws and cancels later phases ($schedule)" for schedule in (:static, :dynamic, :greedy)
        # Fail inside the first group, so the remaining groups still have
        # barriers that the surviving workers would otherwise wait on forever.
        victim = last(first(groups))
        after = Threads.Atomic{Int}(0)
        @test_throws "region failed" Tesserae.partitioned_foreach(bs, Val(schedule);
                                                                epilogue = (_, _) -> Threads.atomic_add!(after, 1)) do region
            region == victim && error("region failed")
            # Neither later groups nor the epilogue may run after a failure.
            !(region in first(groups)) && Threads.atomic_add!(after, 1)
        end
        @test after[] == 0
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
                for bounds in (Tesserae.equal_count_bounds!(Vector{Int}(undef, nworkers + 1), nregions, nworkers),
                               Tesserae.particle_count_bounds!(Vector{Int}(undef, nworkers + 1), bs, group, nworkers))
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

    @testset "hooks surround the regions, including a single-region partition" begin
        single = Tesserae.CPUBlockStrategy(mesh)
        update!(single, [Vec(0.01, 0.01)])
        for (strat, with_prologue) in ((bs, true), (single, false))
            nregions = sum(length, Tesserae.threadsafe_groups(strat))
            initialized = Threads.Atomic{Int}(0)
            visited = Threads.Atomic{Int}(0)
            finished = Threads.Atomic{Int}(0)
            out_of_order = Threads.Atomic{Bool}(false)
            prologue = with_prologue ? (_, _) -> Threads.atomic_add!(initialized, 1) : nothing
            epilogue = (nworkers, w) -> begin
                visited[] == nregions || (out_of_order[] = true)
                Threads.atomic_add!(finished, 1)
            end
            Tesserae.partitioned_foreach(strat, Val(:dynamic); prologue, epilogue) do region
                if with_prologue && initialized[] != Threads.nthreads()
                    out_of_order[] = true
                end
                Threads.atomic_add!(visited, 1)
            end
            @test !out_of_order[]
            @test (visited[], finished[]) == (nregions, Threads.nthreads())
        end
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
