@testset "BlockSpace" begin
    mesh = CartesianMesh(0.2, (0,3), (0,4))
    xₚ = generate_particles(mesh)
    filter!(xₚ) do (x,y)
        (x-1.5)^2 + (y-2)^2 < 1
    end
    blockspace = (@inferred BlockSpace(mesh))
    @test size(blockspace) === blocksize(mesh)
    @test all(isempty, blockspace)
    @test typeof(blockspace[1]) === eltype(blockspace)
    update!(blockspace, xₚ)
    ptsinblks = map(_->Int[], CartesianIndices(blocksize(mesh)))
    for p in eachindex(xₚ)
        I = Sequoia.whichblock(xₚ[p], mesh)
        I === nothing || push!(ptsinblks[I], p)
    end
    @test blockspace == ptsinblks
end
