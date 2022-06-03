@testset "sparsity pattern" begin
    interp = LinearWLS(CubicBSpline())
    grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0)
    pointstate = generate_pointstate((x,y) -> y < 5.0, interp, grid)
    cache = MPCache(interp, grid, pointstate)
    mask = falses(size(grid))
    mask[:,6:end] .= true
    update!(cache, pointstate; exclude = mask)
    @test !any(cache.spat[:,7:end])
end
