@testset "P2G" begin
    transfer = Transfer()
    for interp in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
        for coordinate_system in (PlaneStrain(), Axisymmetric())
            # initialization
            grid = Grid(0.0:2.0:10.0, 0.0:2.0:20.0; coordinate_system)
            gridstate = generate_gridstate(interp, grid)
            pointstate = generate_pointstate((x,y) -> true, interp, grid)
            space = MPSpace(interp, grid, pointstate.x)
            v0 = rand(Vec{2})
            ρ0 = 1.2e3
            @. pointstate.m = ρ0 * pointstate.V
            @. pointstate.v = v0
            @. pointstate.σ = zero(SymmetricSecondOrderTensor{3})
            # transfer
            update!(space, pointstate)
            update_sppattern!(gridstate, space)
            transfer.point_to_grid!(gridstate, pointstate, space, 1)
            @test all(==(v0), pointstate.v)
        end
    end
end

