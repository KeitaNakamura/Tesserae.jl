@testset "Generating point state" begin
    # plane strain
    grid = Grid(0:0.1:10, 0:0.1:10)
    pointstate = generate_pointstate((x,y) -> true, grid)
    @test sum(pointstate.V) ≈ 10*10
    # axisymmetric
    grid = Grid(0:0.1:10, 0:0.1:10; coordinate_system = Axisymmetric())
    pointstate = generate_pointstate((x,y) -> true, grid)
    @test sum(pointstate.V) ≈ 10^2/2 * 10 # 1 radian
end
