@testset "Transfers between grid and particles" begin
    ParticleState = @NamedTuple begin
        m::Float64
        V::Float64
        x::Vec{2, Float64}
        v::Vec{2, Float64}
        b::Vec{2, Float64}
        σ::SymmetricSecondOrderTensor{3, Float64, 6}
        ∇v::SecondOrderTensor{3, Float64, 9}
        B::Mat{2, 2, Float64, 4} # for APIC
        C::Mat{2, 3, Float64, 6} # for LinearWLS
    end
    GridState = @NamedTuple begin
        x::Vec{2, Float64}
        m::Float64
        v::Vec{2, Float64}
        vⁿ::Vec{2, Float64}
    end
    @testset "P2G" begin
        for interp in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
            for system in (PlaneStrain(), Axisymmetric())
                # initialization
                grid = generate_grid(GridState, system, 2.0, (0,10), (0,20))
                particles= generate_particles((x,y) -> true, ParticleState, grid)
                space = MPSpace(interp, grid, particles)
                v0 = rand(Vec{2})
                ρ0 = 1.2e3
                @. particles.m = ρ0 * particles.V
                @. particles.v = v0
                @. particles.σ = zero(SymmetricSecondOrderTensor{3})
                # transfer
                update!(space, grid, particles)
                particles_to_grid!(FLIP(), grid, particles, space, 1)
                @test all(==(v0), particles.v)
            end
        end
    end
    @testset "Check coincidence in LinearWLS/APIC/TPIC" begin # should be identical when LinearWLS interpolation is used except near boundary
        grid = generate_grid(GridState, 0.1, (-10,10), (-10,10))
        grid_v = [rand(x) for x in grid.x]

        for include_near_boundary in (true, false)
            for kernel in (QuadraticBSpline(), CubicBSpline())
                interp = LinearWLS(kernel)
                wls, apic, tpic = map((DefaultTransfer(), APIC(), TPIC())) do transfer
                    dt = 0.002

                    if include_near_boundary
                        particles = generate_particles((x,y) -> true, ParticleState, grid)
                    else
                        particles = generate_particles((x,y) -> -5<x<5 && -5<y<5, ParticleState, grid)
                    end
                    @. particles.m = 1
                    x₀ = copy(particles.x)

                    space = MPSpace(interp, grid, particles)
                    # update interpolation values and sparsity pattern
                    update!(space, grid, particles)

                    # initialize particles
                    grid.v .= grid_v
                    grid_to_particles!(transfer, particles, grid, space, dt)

                    for step in 1:10
                        update!(space, grid, particles)
                        particles_to_grid!(transfer, grid, particles, space, dt)
                        grid_to_particles!(transfer, particles, grid, space, dt)
                    end

                    # check if movement of particles is large enough
                    @test !(isapprox(particles.x, x₀; atol=1.0))

                    [particles.x; particles.v]
                end
                if include_near_boundary
                    @test wls ≈ tpic
                    @test !(apic ≈ tpic)
                    @test !(apic ≈ wls)
                else
                    @test wls ≈ apic ≈ tpic
                end
            end
        end
    end
    @testset "Check preservation of velocity gradient" begin
        grid = generate_grid(GridState, 1.0, (-10,10), (-10,10))
        ∇v = rand(Mat{2,2})
        grid_v = [∇v⋅x for x in grid.x] # create linear distribution

        for include_near_boundary in (true, false,)
            @testset "$kernel" for kernel in (QuadraticBSpline(), CubicBSpline())
                @testset "$interp" for interp in (KernelCorrection(kernel), LinearWLS(kernel))
                    @testset "$transfer" for transfer in (FLIP(), TPIC(), APIC(), DefaultTransfer())
                        interp isa KernelCorrection && transfer isa DefaultTransfer && continue

                        dt = 1.0

                        if include_near_boundary
                            particles = generate_particles((x,y) -> true, ParticleState, grid; random=true)
                        else
                            particles = generate_particles((x,y) -> -5<x<5 && -5<y<5, ParticleState, grid; random=true)
                        end
                        @. particles.m = 1
                        x₀ = copy(particles.x)

                        space = MPSpace(interp, grid, particles)
                        # update interpolation values and sparsity pattern
                        update!(space, grid, particles)

                        # initialize point states
                        grid.v .= grid_v
                        if transfer isa FLIP
                            # use PIC to correctly initialize particle velocity
                            grid_to_particles!(PIC(), particles, grid, space, dt)
                        else
                            grid_to_particles!(transfer, particles, grid, space, dt)
                        end
                        particles.x .= x₀

                        v₀ = copy(particles.v)
                        ∇v₀ = copy(particles.∇v)

                        # outdir = "testdir"
                        # mkpath(outdir)
                        # pvdfile = joinpath(outdir, "test")
                        # closepvd(openpvd(pvdfile))
                        # openpvd(pvdfile; append=true) do pvd
                            # openvtm(string(pvdfile, 0)) do vtm
                                # openvtk(vtm, particles.x) do vtk
                                    # vtk["velocity"] = particles.v
                                    # vtk["velocity gradient"] = vec.(particles.∇v)
                                # end
                                # pvd[0] = vtm
                            # end
                        # end

                        particles_set = map(1:10) do step
                            update!(space, grid, particles)
                            particles_to_grid!(transfer, grid, particles, space, dt)
                            grid_to_particles!(transfer, particles, grid, space, dt)
                            particles.x .= x₀

                            # openpvd(pvdfile; append=true) do pvd
                                # openvtm(string(pvdfile, step)) do vtm
                                    # openvtk(vtm, particles.x) do vtk
                                        # vtk["velocity"] = particles.v
                                        # vtk["velocity gradient"] = vec.(particles.∇v)
                                    # end
                                    # pvd[step] = vtm
                                # end
                            # end

                            particles
                        end

                        for particles in particles_set
                            if transfer isa FLIP
                                # velocity gradient is preserved only if the particle is far from boundary
                                @test !(∇v₀ ≈ particles.∇v)
                                # but there is no change after the first step
                                @test particles.∇v ≈ particles_set[1].∇v
                            else
                                @test v₀ ≈ particles.v
                                @test ∇v₀ ≈ particles.∇v
                            end
                        end
                    end
                end
            end
        end
    end
end
