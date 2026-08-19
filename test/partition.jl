# Components of the same element type share one reorder buffer, so a reorder
# has to finish with one component before starting the next. Two `Vec` fields
# and two scalar fields catch a batching that forgets this.
@testset "reorder_particles! with repeated component types" begin
    mesh = CartesianMesh(0.1, (0,1), (0,1))
    Prop = @NamedTuple begin
        x :: Vec{2, Float64}
        v :: Vec{2, Float64}
        m :: Float64
        V :: Float64
    end
    particles = generate_particles(Prop, mesh)
    for p in eachindex(particles)
        particles.v[p] = 10 * particles.x[p]
        particles.m[p] = p
        particles.V[p] = -p
    end
    before = (x=copy(particles.x), v=copy(particles.v), m=copy(particles.m), V=copy(particles.V))

    bs = Tesserae.CPUBlockStrategy(mesh)
    Random.seed!(1234)
    shuffle!(particles)
    update!(bs, particles.x)
    @test reorder_particles!(particles, bs)

    # Every field must still travel with its own particle.
    @test sort(particles.m) == sort(before.m)
    for p in eachindex(particles)
        @test particles.v[p] ≈ 10 * particles.x[p]
        @test particles.V[p] == -particles.m[p]
    end
end

@testset "_permute_component! gathers correctly at every size" begin
    # Sizes either side of the chunk count, so the last chunk is short, exact,
    # and empty in turn. Which of those a worker gets is a scheduling detail;
    # none of them may change the result.
    Random.seed!(1234)
    nt = Threads.nthreads()
    for T in (Float64, Vec{3,Float64})
        for len in (1, 2, nt - 1, nt, nt + 1, 2nt, 2nt + 3, 1000)
            len < 1 && continue
            component = T[T <: Real ? T(k) : T(ntuple(i -> Float64(10k + i), 3)) for k in 1:len]
            perm = randperm(len)
            buffer = similar(component)
            expected = component[perm]
            Tesserae._permute_component!(component, perm, buffer)
            @test component == expected
        end
    end
end

@testset "Partition" begin
    @testset "CPUBlockStrategy" begin
        mesh = CartesianMesh(0.25, (0,4), (0,4))
        particles = generate_particles(@NamedTuple{x::Vec{2, Float64}}, mesh)
        filter!(particles) do particle
            x, y = particle.x
            (x-2)^2 + (y-2)^2 < 1
        end

        Random.seed!(1234)
        shuffle!(particles)
        xₚ = particles.x

        bs = (@inferred Tesserae.CPUBlockStrategy(mesh))
        partition = (@inferred Partition(mesh))
        @test Tesserae.strategy(partition) isa Tesserae.CPUBlockStrategy
        @test Tesserae.nblocks(bs) === Tesserae.nblocks(mesh)
        @test Tesserae.block_size_log2(bs) === Tesserae.block_size_log2(mesh)
        @test Tesserae.blockwidth(bs) === Tesserae.blockwidth(mesh)
        @test all(blk -> isempty(Tesserae.particle_indices(bs, blk)), LinearIndices(Tesserae.nblocks(bs)))
        function check_group_order(bs)
            ordered = Int[]
            for group in Tesserae.threadsafe_groups(bs), blk in group
                append!(ordered, Tesserae.particle_indices(bs, blk))
            end
            @test ordered == bs.particleindices[1:Tesserae.nassigned(bs)]
        end
        function check_particle_blocks(bs, mesh, xₚ)
            expected = map(_ -> Int[], CartesianIndices(Tesserae.nblocks(mesh)))
            n_assigned = 0
            for p in eachindex(xₚ)
                I = Tesserae.findblock(xₚ[p], mesh)
                if I !== nothing
                    push!(expected[I], p)
                    n_assigned += 1
                end
            end
            actual = map(blk -> collect(Tesserae.particle_indices(bs, blk)), LinearIndices(Tesserae.nblocks(bs)))
            @test actual == expected
            @test Tesserae.nassigned(bs) == n_assigned
        end
        update!(bs, xₚ)
        check_group_order(bs)
        check_particle_blocks(bs, mesh, xₚ)
        basis = BSpline(Cubic())
        for group in Tesserae.threadsafe_groups(bs)
            group_nodes = Set{CartesianIndex{2}}()
            for blk in group
                block_nodes = Set{CartesianIndex{2}}()
                for p in Tesserae.particle_indices(bs, blk)
                    union!(block_nodes, Tesserae.supportnodes(basis, xₚ[p], mesh))
                end
                @test isempty(intersect(group_nodes, block_nodes))
                union!(group_nodes, block_nodes)
            end
        end

        # Reordering should keep block ranges and P2G color-group order valid.
        @test reorder_particles!(particles, bs)

        n_assigned = Tesserae.nassigned(bs)
        @test bs.particleindices[1:n_assigned] == collect(1:n_assigned)
        @test Tesserae.block_ordered_particle_contiguity(bs) == 1.0
        @test !reorder_particles!(particles, bs; threshold=0.85)
        @test reorder_particles!(particles, bs; threshold=1.0)
        check_group_order(bs)

        update!(bs, xₚ)
        check_group_order(bs)
        @test !reorder_particles!(particles, bs; threshold=0.85)
        n_assigned = Tesserae.nassigned(bs)
        @test bs.particleindices[1:n_assigned] == collect(1:n_assigned)
        check_group_order(bs)

        check_particle_blocks(bs, mesh, xₚ)

        moving_bs = Tesserae.CPUBlockStrategy(mesh)
        moving_xₚ = [Vec(0.125, 0.125), Vec(0.375, 0.375), Vec(3.625, 3.625), Vec(3.875, 3.875)]
        update!(moving_bs, moving_xₚ)
        check_group_order(moving_bs)
        check_particle_blocks(moving_bs, mesh, moving_xₚ)

        moving_xₚ = [Vec(0.125, 0.125), Vec(0.375, 0.375)]
        update!(moving_bs, moving_xₚ)
        check_group_order(moving_bs)
        check_particle_blocks(moving_bs, mesh, moving_xₚ)

        moving_particles = generate_particles(@NamedTuple{x::Vec{2, Float64}}, mesh)
        resize!(moving_particles, 3)
        moving_particles.x .= [Vec(0.125, 0.125), Vec(10.0, 10.0), Vec(3.875, 0.125)]
        moving_xₚ = moving_particles.x
        update!(moving_bs, moving_xₚ)
        check_group_order(moving_bs)
        check_particle_blocks(moving_bs, mesh, moving_xₚ)
        @test Tesserae.block_ordered_particle_contiguity(moving_bs) == 0.0
        @test !reorder_particles!(moving_particles, moving_bs; threshold=0.0)
        @test_logs (:warn, r"Some particles are outside of the grid") begin
            @test reorder_particles!(moving_particles, moving_bs; threshold=0.5)
        end
        n_assigned = Tesserae.nassigned(moving_bs)
        @test moving_bs.particleindices[1:n_assigned] == collect(1:n_assigned)
        @test Tesserae.block_ordered_particle_contiguity(moving_bs) == 1.0
        @test Tesserae.findblock(moving_xₚ[end], mesh) === nothing
        check_group_order(moving_bs)
        check_particle_blocks(moving_bs, mesh, moving_xₚ)
    end
    @testset "CellStrategy" begin
        mesh = FEMesh(CartesianMesh(0.5, (0,2), (0,2)))
        partition = Partition(mesh)
        strat = Tesserae.strategy(partition)
        groups = Tesserae.threadsafe_groups(strat)
        @test all(!isempty, groups)
        for group in groups
            for i in 1:length(group)-1, j in i+1:length(group)
                cell1, cell2 = group[i], group[j]
                nodes1 = supportnodes(mesh, cell1)
                nodes2 = supportnodes(mesh, cell2)
                @test isempty(intersect(Set(nodes1), Set(nodes2)))
            end
        end
        allcells = reduce(vcat, groups)
        @test length(allcells) == Tesserae.ncells(mesh)
        @test sort(allcells) == collect(1:Tesserae.ncells(mesh))
        @test collect(Tesserae.particle_indices(partition, zeros(3, Tesserae.ncells(mesh)), first(first(groups)))) ==
              [CartesianIndex(p, first(first(groups))) for p in 1:3]

        iga_mesh = IGAMesh(CartesianMesh(0.25, (0,2), (0,2)); degree=Quadratic())
        iga_partition = @inferred Partition(iga_mesh)
        iga_groups = Tesserae.threadsafe_groups(iga_partition)
        iga_cells = collect(cells(iga_mesh))
        @test all(!isempty, iga_groups)
        for group in iga_groups
            for i in 1:length(group)-1, j in i+1:length(group)
                nodes1 = supportnodes(iga_mesh, iga_cells[group[i]])
                nodes2 = supportnodes(iga_mesh, iga_cells[group[j]])
                @test isempty(intersect(Set(nodes1), Set(nodes2)))
            end
        end
        iga_allcells = reduce(vcat, iga_groups)
        @test length(iga_allcells) == Tesserae.ncells(iga_mesh)
        @test sort(iga_allcells) == collect(1:Tesserae.ncells(iga_mesh))
        @test collect(Tesserae.particle_indices(iga_partition, zeros(3, Tesserae.ncells(iga_mesh)), first(first(iga_groups)))) ==
              [CartesianIndex(p, first(first(iga_groups))) for p in 1:3]
    end
    @testset "Utilities" begin
        mesh = CartesianMesh(1, (0, 20), (0, 20); block_size_log2=Val(3))
        @test Tesserae.nblocks(mesh) === (3, 3)
    end
    # The device tags dispatch without a GPU, so the contract errors on the
    # device/partition boundary are testable on CPU.
    @testset "device and partition mismatch errors" begin
        mesh = CartesianMesh(1.0, (0,8), (0,8))
        cpu_partition = Partition(mesh)
        gpu_partition = Tesserae.Partition(Tesserae.GPUBlockStrategy(mesh))
        gpu_device = Tesserae.CUDADevice{Tesserae.CastFloat32}()
        grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)
        particles = generate_particles(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)
        weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))

        @test_throws "CPU-only" update_sparsity!(Tesserae.SpIndices(mesh), gpu_partition)

        @test_throws "lives on the GPU" Tesserae.check_partition_for_transfer("@P2G", Tesserae.CPUDevice(), grid, weights, gpu_partition)
        @test_throws "lives on the CPU" Tesserae.check_partition_for_transfer("@P2G", gpu_device, grid, weights, cpu_partition)
        @test_throws "only supports @P2G" Tesserae.check_partition_for_transfer("@G2P2G", gpu_device, grid, weights, gpu_partition)
        @test_throws "No particles assigned" Tesserae.check_partition_for_transfer("@P2G", gpu_device, grid, weights, gpu_partition)
        @test_throws "No particles assigned" Tesserae.check_partition_for_transfer("@P2G", Tesserae.CPUDevice(), grid, weights, cpu_partition)
    end
    @testset "reorder_particles! on a GPU partition" begin
        mesh = CartesianMesh(1.0, (0,16), (0,16))
        bs = Tesserae.GPUBlockStrategy(mesh)
        gpu_partition = Tesserae.Partition(bs)
        particles = generate_particles(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)
        particles.m .= 1:length(particles)
        shuffled = particles[shuffle(Xoshiro(0), 1:length(particles))]
        nₚ = length(shuffled)

        # The device counting sort's scan kernels carry `@synchronize` inside
        # loops, which the KernelAbstractions CPU backend cannot execute, so
        # the block-sorted state is built directly here; the sort itself is
        # exercised on real GPU backends.
        LI = LinearIndices(Tesserae.nblocks(mesh))
        blkof(x) = LI[Tesserae.findblock(x, mesh)]
        counts = zeros(Int32, length(LI))
        foreach(x -> counts[blkof(x)] += 1, shuffled.x)
        resize!(bs.particleindices, nₚ)
        resize!(bs.blockids, nₚ)
        bs.offsets .= [Int32(0); cumsum(counts)]
        bs.nactive[] = count(!iszero, counts)
        bs.nassigned[] = nₚ
        bs.particleindices .= Int32.(sortperm(map(blkof, shuffled.x)))
        bs.blockids .= Int32.(map(blkof, shuffled.x))

        c₀ = Tesserae.block_ordered_particle_contiguity(gpu_partition)
        @test 0 ≤ c₀ < 0.5
        @test reorder_particles!(shuffled, gpu_partition)
        @test Tesserae.block_ordered_particle_contiguity(gpu_partition) == 1.0

        @test issorted(map(blkof, shuffled.x))
        @test collect(bs.particleindices) == 1:nₚ
        # Components were permuted together: every row still pairs its original x and m.
        @test all(i -> shuffled.x[i] == particles.x[Int(shuffled.m[i])], eachindex(shuffled))
        @test sort(shuffled.m) == 1.0:nₚ

        # Offsets shift globally whenever an upstream block changes its count;
        # the grouping score must not decay from that alone.
        bs.offsets[2:end-1] .+= Int32(1)
        @test Tesserae.block_ordered_particle_contiguity(gpu_partition) == 1.0
        bs.offsets[2:end-1] .-= Int32(1)

        @test !reorder_particles!(shuffled, gpu_partition; threshold=0)
        @test !reorder_particles!(shuffled, gpu_partition; threshold=0.5)
        @test_throws ArgumentError reorder_particles!(shuffled, gpu_partition; threshold=1.5)

        # Empty particles reorder as a no-op even against stale nonzero buffers
        # (Metal keeps the old buffer length when resized to empty).
        @test reorder_particles!(shuffled[1:0], gpu_partition)

        bs.offsets .= Int32(0)
        bs.offsets[end] = Int32(nₚ - 1)
        bs.nassigned[] = nₚ - 1
        @test_throws "outside the mesh" reorder_particles!(shuffled, gpu_partition)
    end
end
