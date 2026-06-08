@testset "ThreadPartition" begin
    @testset "BlockStrategy" begin
        mesh = CartesianMesh(0.25, (0,4), (0,4))
        particles = generate_particles(@NamedTuple{x::Vec{2, Float64}}, mesh)
        filter!(particles) do particle
            x, y = particle.x
            (x-2)^2 + (y-2)^2 < 1
        end

        Random.seed!(1234)
        shuffle!(particles)
        xₚ = particles.x

        bs = (@inferred Tesserae.BlockStrategy(mesh))
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

        moving_bs = Tesserae.BlockStrategy(mesh)
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
        mesh = UnstructuredMesh(CartesianMesh(0.5, (0,2), (0,2)))
        partition = ThreadPartition(mesh)
        strat = Tesserae.strategy(partition)
        groups = Tesserae.threadsafe_groups(strat)
        @test all(!isempty, groups)
        for group in groups
            for i in 1:length(group)-1, j in i+1:length(group)
                cell1, cell2 = group[i], group[j]
                nodes1 = Tesserae.cellnodeindices(mesh, cell1)
                nodes2 = Tesserae.cellnodeindices(mesh, cell2)
                @test isempty(intersect(Set(nodes1), Set(nodes2)))
            end
        end
        allcells = reduce(vcat, groups)
        @test length(allcells) == Tesserae.ncells(mesh)
        @test sort(allcells) == collect(1:Tesserae.ncells(mesh))
        @test collect(Tesserae.particle_indices(partition, zeros(3, Tesserae.ncells(mesh)), first(first(groups)))) ==
              [CartesianIndex(p, first(first(groups))) for p in 1:3]
    end
    @testset "Utilities" begin
        @test Tesserae.nodeindices_in_block(CartesianIndex(1,1), (20,20); block_size_log2=Val(2)) === CartesianIndices((1:5,1:5))
        @test Tesserae.nodeindices_in_block(CartesianIndex(2,3), (20,20); block_size_log2=Val(2)) === CartesianIndices((5:9,9:13))
        @test Tesserae.nodeindices_in_block(CartesianIndex(2,3), (10,10); block_size_log2=Val(2)) === CartesianIndices((5:9,9:10))
        @test_throws BoundsError Tesserae.nodeindices_in_block(CartesianIndex(2,3), (5,5); block_size_log2=Val(2))

        mesh = CartesianMesh(1, (0, 20), (0, 20); block_size_log2=Val(3))
        @test Tesserae.nblocks(mesh) === (3, 3)
        @test Tesserae.nodeindices_in_block(CartesianIndex(2,2), mesh) === CartesianIndices((9:17,9:17))
    end
end
