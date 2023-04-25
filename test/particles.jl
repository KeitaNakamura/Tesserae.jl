@testset "Generating particles" begin
    @testset "generation (alg=$alg)" for alg in (GridSampling(), PoissonDiskSampling())
        # plane strain
        n = 3
        lattice = Lattice(0.1, (0,10), (0,10))
        particles = generate_particles((x,y) -> true, lattice; alg, spacing=1/n)
        @test sum(particles.V) ≈ 10*10
        @test all(pt->pt.l==spacing(lattice)/n, particles)
        # axisymmetric
        lattice = Lattice(0.1, (0,10), (0,10))
        particles = generate_particles((x,y) -> true, lattice; alg, system=Axisymmetric())
        if alg isa GridSampling
            @test sum(particles.V) ≈ 10^2/2 * 10 # 1 radian
        elseif alg isa PoissonDiskSampling
            @test sum(particles.V) ≈ 10^2/2 * 10 rtol=1e-2 # 1 radian
        else
            error("unreachable")
        end
    end
    @testset "Specify RNG" begin
        lattice = Lattice(0.1, (0,10), (0,10))
        particles = generate_particles((x,y) -> true, lattice; alg=PoissonDiskSampling(StableRNG(1234)))
        @test mean(particles.x) ≈ [5,5] rtol=0.01
    end
end
