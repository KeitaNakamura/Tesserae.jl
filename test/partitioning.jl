@testset "ColorPartition" begin
    @testset "BlockStrategy" begin
        mesh = CartesianMesh(0.2, (0,3), (0,4))
        xₚ = generate_particles(mesh)
        filter!(xₚ) do (x,y)
            (x-1.5)^2 + (y-2)^2 < 1
        end
        bs = (@inferred Tesserae.BlockStrategy(mesh))
        @test Tesserae.blocksize(bs) === Tesserae.blocksize(mesh)
        @test all(blk -> isempty(Tesserae.particle_indices_in(bs, blk)), Tesserae.blockindices(bs))
        update!(bs, xₚ)
        ptsinblks = map(_->Int[], CartesianIndices(Tesserae.blocksize(mesh)))
        for p in eachindex(xₚ)
            I = Tesserae.whichblock(xₚ[p], mesh)
            I === nothing || push!(ptsinblks[I], p)
        end
        @test map(blk -> Tesserae.particle_indices_in(bs, blk), Tesserae.blockindices(bs)) == ptsinblks
    end
end
