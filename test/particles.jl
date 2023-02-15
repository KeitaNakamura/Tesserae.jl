@testset "Generating particles" begin
    @testset "generation (random=$random)" for random in (false, true)
        # plane strain
        lattice = Lattice(0.1, (0,10), (0,10))
        particles = generate_particles((x,y) -> true, lattice; random)
        @test sum(particles.V) ≈ 10*10
        @test all(particles) do pt
            (pt.l)^2 ≈ pt.V
        end
        # axisymmetric
        lattice = Lattice(0.1, (0,10), (0,10))
        particles = generate_particles((x,y) -> true, lattice; random, system=Axisymmetric())
        if random == false
            @test sum(particles.V) ≈ 10^2/2 * 10 # 1 radian
        else
            @test sum(particles.V) ≈ 10^2/2 * 10 rtol=1e-2 # 1 radian
        end
    end
    @testset "copied from existing particles" begin
        lattice = Lattice(0.1, (0,10), (0,10))
        particles_old = generate_particles((x,y) -> true, lattice; random=true)
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
