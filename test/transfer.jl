@testset "Transfer" begin
    @testset "P2G" begin
        PointState = @NamedTuple begin
            m::Float64
            V::Float64
            x::Vec{2, Float64}
            v::Vec{2, Float64}
            b::Vec{2, Float64}
            σ::SymmetricSecondOrderTensor{3, Float64, 6}
        end
        GridState = @NamedTuple begin
            m::Float64
            v::Vec{2, Float64}
            v_n::Vec{2, Float64}
        end
        for interp in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
            transfer = DefaultTransfer(interp)
            for coordinate_system in (PlaneStrain(), Axisymmetric())
                # initialization
                grid = Grid(0.0:2.0:10.0, 0.0:2.0:20.0; coordinate_system)
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
    @testset "LinearWLS/APIC/TPIC" begin # should be identical when LinearWLS interpolation is used except near boundary
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
            v_n::Vec{2, Float64}
        end

        grid = Grid(-10:0.1:10, -10:0.1:10)
        gridstate = generate_gridstate(GridState, grid)
        grid_v = [rand(Vec{2}) for _ in grid]

        for include_near_boundary in (true, false)
            for kernel in (QuadraticBSpline(), CubicBSpline())
                interp = LinearWLS(kernel)
                wls, apic, tpic = map((DefaultTransfer(interp), APIC(), TPIC())) do transfer
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
end
