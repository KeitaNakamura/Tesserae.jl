@testset "SpSpace" begin
    mesh = CartesianMesh(0.2, (0,3), (0,4))
    xₚ = generate_particles(mesh).x
    filter!(xₚ) do (x,y)
        (x-1.5)^2 + (y-2)^2 < 1
    end
    spspace = (@inferred SpSpace(mesh))
    @test size(spspace) === blocksize(mesh)
    @test all(isempty, spspace)
    @test typeof(spspace[1]) === eltype(spspace)
    update!(spspace, xₚ)
    ptsinblks = map(_->Int[], CartesianIndices(blocksize(mesh)))
    for p in eachindex(xₚ)
        I = Sequoia.whichblock(xₚ[p], mesh)
        I === nothing || push!(ptsinblks[I], p)
    end
    @test spspace == ptsinblks
end
