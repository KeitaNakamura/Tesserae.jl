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
            transfer = DefaultTransfer()
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
end
