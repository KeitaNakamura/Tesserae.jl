@testset "MPSpace" begin
    @testset "Sparsity pattern" begin
        interp = LinearWLS(CubicBSpline())
        grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0)
        pointstate = generate_pointstate((x,y) -> y < 5.0, grid)
        space = MPSpace(interp, grid, pointstate)
        filter = trues(size(grid))
        filter[:,6:end] .= false
        update!(space, pointstate; filter)
        @test !any(space.sppat[:,7:end])
    end
end
