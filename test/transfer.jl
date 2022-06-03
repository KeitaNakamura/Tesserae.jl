@testset "P2G" begin
    transfer = Transfer()
    for interp in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
        for coordinate_system in (PlaneStrain(), Axisymmetric())
            # initialization
            grid = Grid(0.0:2.0:10.0, 0.0:2.0:20.0; coordinate_system)
            gridstate = generate_gridstate(interp, grid)
            pointstate = generate_pointstate((x,y) -> true, interp, grid)
            cache = MPCache(interp, grid, pointstate.x)
            v0 = rand(Vec{2})
            ρ0 = 1.2e3
            @. pointstate.m = ρ0 * pointstate.V
            @. pointstate.v = v0
            @. pointstate.σ = zero(SymmetricSecondOrderTensor{3})
            # transfer
            update!(cache, pointstate)
            update_sparsitypattern!(gridstate, cache)
            transfer.point_to_grid!(gridstate, pointstate, cache, 1)
            @test all(==(v0), pointstate.v)
        end
    end
end

