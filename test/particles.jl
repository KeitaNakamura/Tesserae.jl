@testset "Generating particles" begin
    @testset "generation (alg=$alg)" for alg in (GridSampling(), PoissonDiskSampling())
        # plane strain
        lattice = Lattice(0.1, (0,10), (0,10))
        particles = generate_particles((x,y) -> true, lattice; alg)
        @test sum(particles.V) ≈ 10*10
        @test all(particles) do pt
            (pt.l)^2 ≈ pt.V
        end
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
        @test mean(particles.x) ≈ [4.985641459718793, 4.9856469624835285]
    end
    @testset "copied from existing particles" begin
        lattice = Lattice(0.1, (0,10), (0,10))
        particles_old = generate_particles((x,y) -> true, lattice; alg=PoissonDiskSampling())
        check_particles = particles_new -> begin
            @test particles_new !== particles_old
            @test particles_new.x !== particles_old.x
            @test particles_new.V !== particles_old.V
            @test particles_new.l !== particles_old.l
            @test particles_new == particles_old
            @test particles_new.x == particles_old.x
            @test particles_new.V == particles_old.V
            @test particles_new.l == particles_old.l
        end
        check_particles(@inferred generate_particles(eltype(particles_old), particles_old))
        check_particles(@inferred generate_particles(particles_old))
    end
end
