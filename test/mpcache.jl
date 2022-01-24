@testset "P2G" begin
    for it in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
        for coordinate_system in (PlaneStrain(), Axisymmetric())
            # initialization
            grid = Grid(it, 0.0:2.0:10.0, 0.0:2.0:20.0; coordinate_system)
            pointstate = generate_pointstate((x,y) -> true, grid)
            cache = MPCache(grid, pointstate.x)
            v0 = rand(Vec{2})
            ρ0 = 1.2e3
            @. pointstate.m = ρ0 * pointstate.V
            @. pointstate.v = v0
            @. pointstate.σ = zero(SymmetricSecondOrderTensor{3})
            # transfer
            update!(cache, grid, pointstate)
            default_point_to_grid!(grid, pointstate, cache, 1)
            @test all(==(v0), pointstate.v)
        end
    end
end

@testset "sparsity pattern" begin
    grid = Grid(LinearWLS(CubicBSpline()), 0.0:1.0:10.0, 0.0:1.0:10.0)
    pointstate = generate_pointstate((x,y) -> y < 5.0, grid)
    cache = MPCache(grid, pointstate)
    mask = falses(size(grid))
    mask[:,6:end] .= true
    update!(cache, grid, pointstate; exclude = mask)
    @test !any(cache.spat[:,7:end])
end
