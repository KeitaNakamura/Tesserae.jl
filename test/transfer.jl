@testset "Transfers between grid and particles" begin
    PointState = @NamedTuple begin
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
        m::Float64
        v::Vec{2, Float64}
        vⁿ::Vec{2, Float64}
    end
    @testset "P2G" begin
        for interp in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
            transfer = Transfer(interp)
            for system in (PlaneStrain(), Axisymmetric())
                # initialization
                grid = Grid(system, 0.0:2.0:10.0, 0.0:2.0:20.0)
                gridstate = generate_gridstate(GridState, grid)
                pointstate = generate_pointstate((x,y) -> true, PointState, grid)
                space = MPSpace(interp, grid, pointstate.x)
                v0 = rand(Vec{2})
                ρ0 = 1.2e3
                @. pointstate.m = ρ0 * pointstate.V
                @. pointstate.v = v0
                @. pointstate.σ = zero(SymmetricSecondOrderTensor{3})
                # transfer
                update!(space, pointstate)
                update_sparsity_pattern!(gridstate, space)
                point_to_grid!(transfer, gridstate, pointstate, space, 1)
                @test all(==(v0), pointstate.v)
            end
        end
    end
    @testset "Check coincidence in LinearWLS/APIC/TPIC" begin # should be identical when LinearWLS interpolation is used except near boundary
        grid = Grid(-10:0.1:10, -10:0.1:10)
        gridstate = generate_gridstate(GridState, grid)
        grid_v = [rand(Vec{2}) for _ in grid]

        for include_near_boundary in (true, false)
            for kernel in (QuadraticBSpline(), CubicBSpline())
                interp = LinearWLS(kernel)
                wls, apic, tpic = map((Transfer(interp), APIC(), TPIC())) do transfer
                    dt = 0.002

                    if include_near_boundary
                        pointstate = generate_pointstate((x,y) -> true, PointState, grid)
                    else
                        pointstate = generate_pointstate((x,y) -> -5<x<5 && -5<y<5, PointState, grid)
                    end
                    @. pointstate.m = 1
                    x₀ = copy(pointstate.x)

                    space = MPSpace(interp, grid, pointstate.x)
                    # update interpolation values and sparsity pattern
                    update!(space, pointstate)
                    update_sparsity_pattern!(gridstate, space)

                    # initialize point states
                    gridstate.v .= grid_v
                    grid_to_point!(transfer, pointstate, gridstate, space, dt)

                    for step in 1:10
                        update!(space, pointstate)
                        update_sparsity_pattern!(gridstate, space)
                        point_to_grid!(transfer, gridstate, pointstate, space, dt)
                        grid_to_point!(transfer, pointstate, gridstate, space, dt)
                    end

                    # check if movement of particles is large enough
                    @test !(isapprox(pointstate.x, x₀; atol=1.0))

                    [pointstate.x; pointstate.v]
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
        grid = Grid(-10:1.0:10, -10:1.0:10)
        gridstate = generate_gridstate(GridState, grid)
        ∇v = rand(Mat{2,2})
        grid_v = [∇v⋅x for x in grid] # create linear distribution

        for include_near_boundary in (true, false,)
            for kernel in (QuadraticBSpline(), CubicBSpline())
                for interp in (KernelCorrection(kernel), LinearWLS(kernel))
                    for transfer in (FLIP(), TPIC(), APIC(), Transfer(interp))
                        dt = 1.0

                        if include_near_boundary
                            pointstate = generate_pointstate((x,y) -> true, PointState, grid; random=true)
                        else
                            pointstate = generate_pointstate((x,y) -> -5<x<5 && -5<y<5, PointState, grid; random=true)
                        end
                        @. pointstate.m = 1
                        x₀ = copy(pointstate.x)

                        space = MPSpace(interp, grid, pointstate.x)
                        # update interpolation values and sparsity pattern
                        update!(space, pointstate)
                        update_sparsity_pattern!(gridstate, space)

                        # initialize point states
                        gridstate.v .= grid_v
                        if transfer isa FLIP
                            # use PIC to correctly initialize particle velocity
                            grid_to_point!(PIC(), pointstate, gridstate, space, dt)
                        else
                            grid_to_point!(transfer, pointstate, gridstate, space, dt)
                        end
                        pointstate.x .= x₀

                        v₀ = copy(pointstate.v)
                        ∇v₀ = copy(pointstate.∇v)

                        # outdir = "testdir"
                        # mkpath(outdir)
                        # pvdfile = joinpath(outdir, "test")
                        # closepvd(openpvd(pvdfile))
                        # openpvd(pvdfile; append=true) do pvd
                            # openvtm(string(pvdfile, 0)) do vtm
                                # openvtk(vtm, pointstate.x) do vtk
                                    # vtk["velocity"] = pointstate.v
                                    # vtk["velocity gradient"] = vec.(pointstate.∇v)
                                # end
                                # pvd[0] = vtm
                            # end
                        # end

                        pointstates = map(1:10) do step
                            update!(space, pointstate)
                            update_sparsity_pattern!(gridstate, space)
                            point_to_grid!(transfer, gridstate, pointstate, space, dt)
                            grid_to_point!(transfer, pointstate, gridstate, space, dt)
                            pointstate.x .= x₀

                            # openpvd(pvdfile; append=true) do pvd
                                # openvtm(string(pvdfile, step)) do vtm
                                    # openvtk(vtm, pointstate.x) do vtk
                                        # vtk["velocity"] = pointstate.v
                                        # vtk["velocity gradient"] = vec.(pointstate.∇v)
                                    # end
                                    # pvd[step] = vtm
                                # end
                            # end

                            pointstate
                        end

                        for pointstate in pointstates
                            if transfer isa FLIP
                                # velocity gradient is preserved only if the particle is far from boundary
                                @test !(∇v₀ ≈ pointstate.∇v)
                                # but there is no change after the first step
                                @test pointstate.∇v ≈ pointstates[1].∇v
                            else
                                @test v₀ ≈ pointstate.v
                                @test ∇v₀ ≈ pointstate.∇v
                            end
                        end
                    end
                end
            end
        end
    end
end
