@testset "sparsity pattern" begin
    grid = Grid(LinearWLS(CubicBSpline()), 0.0:1.0:10.0, 0.0:1.0:10.0)
    pointstate = generate_pointstate((x,y) -> y < 5.0, grid)
    cache = MPCache(grid, pointstate)
    mask = falses(size(grid))
    mask[:,6:end] .= true
    update!(cache, grid, pointstate; exclude = mask)
    @test !any(cache.spat[:,7:end])
end
