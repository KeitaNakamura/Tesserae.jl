@testset "ColorPartition" begin
    @testset "BlockStrategy" begin
        mesh = CartesianMesh(0.2, (0,3), (0,4))
        xₚ = generate_particles(mesh)
        filter!(xₚ) do (x,y)
            (x-1.5)^2 + (y-2)^2 < 1
        end
        bs = (@inferred Tesserae.BlockStrategy(mesh))
        @test Tesserae.nblocks(bs) === Tesserae.nblocks(mesh)
        @test all(blk -> isempty(Tesserae.particle_indices_in(bs, blk)), LinearIndices(Tesserae.nblocks(bs)))
        update!(bs, xₚ)
        ptsinblks = map(_->Int[], CartesianIndices(Tesserae.nblocks(mesh)))
        for p in eachindex(xₚ)
            I = Tesserae.whichblock(xₚ[p], mesh)
            I === nothing || push!(ptsinblks[I], p)
        end
        @test map(blk -> Tesserae.particle_indices_in(bs, blk), LinearIndices(Tesserae.nblocks(bs))) == ptsinblks
    end
    @testset "CellStrategy" begin
        mesh = UnstructuredMesh(CartesianMesh(0.2, (0,3), (0,4)))
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

        # reorder_particles!
        nrow = 3
        ncells = 6
        particles = [100*j + i for i in 1:nrow, j in 1:ncells]
        particles_old = copy(particles)

        groups = [Int[2, 5], Int[1, 4, 6], Int[3]]
        cs = Tesserae.CellStrategy(deepcopy(groups))
        perm = vcat(groups...)
        reorder_particles!(particles, cs)
        @test particles == particles_old[:, perm]
        lens = length.(groups)
        start = 1
        for (gi, len) in pairs(lens)
            @test cs.colorgroups[gi] == collect(start:start+len-1)
            start += len
        end
        @test sort(vcat(cs.colorgroups...)) == collect(1:ncells)
    end
    @testset "Utilities" begin
        @test Tesserae.nodes_in_block(CartesianIndex(1,1), (20,20)) === CartesianIndices((1:5,1:5))
        @test Tesserae.nodes_in_block(CartesianIndex(2,3), (20,20)) === CartesianIndices((5:9,9:13))
        @test Tesserae.nodes_in_block(CartesianIndex(2,3), (10,10)) === CartesianIndices((5:9,9:10))
        @test_throws BoundsError Tesserae.nodes_in_block(CartesianIndex(2,3), (5,5))
    end
end
