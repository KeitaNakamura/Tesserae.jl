@testset "ColorPartition" begin
    @testset "BlockStrategy" begin
        mesh = CartesianMesh(0.25, (0,4), (0,4))
        xₚ = generate_particles(mesh)
        filter!(xₚ) do (x,y)
            (x-2)^2 + (y-2)^2 < 1
        end

        Random.seed!(1234)
        shuffle!(xₚ)

        bs = (@inferred Tesserae.BlockStrategy(mesh))
        @test Tesserae.nblocks(bs) === Tesserae.nblocks(mesh)
        @test Tesserae.block_size_log2(bs) === Tesserae.block_size_log2(mesh)
        @test Tesserae.blockwidth(bs) === Tesserae.blockwidth(mesh)
        @test all(blk -> isempty(Tesserae.particle_indices_in(bs, blk)), LinearIndices(Tesserae.nblocks(bs)))
        update!(bs, xₚ)
        ptsinblks = map(_->Int[], CartesianIndices(Tesserae.nblocks(mesh)))
        for p in eachindex(xₚ)
            I = Tesserae.findblock(xₚ[p], mesh)
            I === nothing || push!(ptsinblks[I], p)
        end
        @test map(blk -> Tesserae.particle_indices_in(bs, blk), LinearIndices(Tesserae.nblocks(bs))) == ptsinblks

        # check reorder_particles
        reorder_particles!(xₚ, bs)

        n_assigned = bs.stops[end]
        @test bs.particleindices[1:n_assigned] == collect(1:n_assigned)

        ptsinblks_after = map(_->Int[], CartesianIndices(Tesserae.nblocks(mesh)))
        for p in eachindex(xₚ)
            I = Tesserae.findblock(xₚ[p], mesh)
            I === nothing || push!(ptsinblks_after[I], p)
        end
        @test map(blk -> Tesserae.particle_indices_in(bs, blk), LinearIndices(Tesserae.nblocks(bs))) == ptsinblks_after
    end
    @testset "CellStrategy" begin
        mesh = UnstructuredMesh(CartesianMesh(0.5, (0,2), (0,2)))
        strat = Tesserae.strategy(ColorPartition(mesh))
        groups = strat.colorgroups
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
