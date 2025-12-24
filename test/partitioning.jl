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
    @testset "Utilities" begin
        @test Tesserae.nodes_in_block(CartesianIndex(1,1), (20,20)) === CartesianIndices((1:5,1:5))
        @test Tesserae.nodes_in_block(CartesianIndex(2,3), (20,20)) === CartesianIndices((5:9,9:13))
        @test Tesserae.nodes_in_block(CartesianIndex(2,3), (10,10)) === CartesianIndices((5:9,9:10))
        @test_throws BoundsError Tesserae.nodes_in_block(CartesianIndex(2,3), (5,5))
    end
end
