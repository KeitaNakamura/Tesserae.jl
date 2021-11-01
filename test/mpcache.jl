@testset "P2G" begin
    for it in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
        for coordinate_system in (:plane_strain, :axisymmetric)
            # initialization
            grid = Grid(it, 0.0:2.0:10.0, 0.0:2.0:20.0; coordinate_system)
            pointstate = generate_pointstate((x,y) -> true, grid)
            cache = MPCache(grid, pointstate.x)
            v0 = rand(Vec{2})
            ρ0 = 1.2e3
            @. pointstate.m = ρ0 * pointstate.V
            @. pointstate.v = v0
            @. pointstate.σ = one(SymmetricSecondOrderTensor{3})
            # transfer
            update!(cache, grid, pointstate)
            default_point_to_grid!(grid, pointstate, cache)
            @test all(==(v0), pointstate.v)
        end
    end
end
