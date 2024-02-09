@testset "SpSpace" begin
    lattice = Lattice(0.2, (0,3), (0,4))
    xₚ = generate_particles(lattice).x
    filter!(xₚ) do (x,y)
        (x-1.5)^2 + (y-2)^2 < 1
    end
    spspace = (@inferred SpSpace(lattice))
    @test size(spspace) === blocksize(lattice)
    @test all(isempty, spspace)
    @test typeof(spspace[1]) === eltype(spspace)
    update!(spspace, xₚ)
    ptsinblks = map(_->Int[], CartesianIndices(blocksize(lattice)))
    for p in eachindex(xₚ)
        I = Sequoia.whichblock(xₚ[p], lattice)
        I === nothing || push!(ptsinblks[I], p)
    end
    @test spspace == ptsinblks
end
