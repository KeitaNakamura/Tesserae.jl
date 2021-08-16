@testset "Grid" begin
    @test @inferred(Grid(0:10))::Grid{1, Int} == Vec.(0:10)
    @test @inferred(Grid(0:10, 0:20))::Grid{2, Int} == Vec.(collect(Iterators.product(0:10, 0:20)))
    @test @inferred(Grid(0:10, 0:20, 0:30))::Grid{3, Int} == Vec.(collect(Iterators.product(0:10, 0:20, 0:30)))
    @test @inferred(Grid{1}(0:10))::Grid{1, Int} == Vec.(0:10)
    @test @inferred(Grid{2}(0:10))::Grid{2, Int} == Vec.(collect(Iterators.product(0:10, 0:10)))
    @test @inferred(Grid{3}(0:10))::Grid{3, Int} == Vec.(collect(Iterators.product(0:10, 0:10, 0:10)))

    grid = Grid(0:1.0:10, 1:2.0:20)
    @test @inferred(gridsteps(grid)) === (1.0, 2.0)
    @test @inferred(gridsteps(grid, 1)) === 1.0
    @test @inferred(gridsteps(grid, 2)) === 2.0
    @test @inferred(gridaxes(grid)) === (0:1.0:10, 1:2.0:20)
    @test @inferred(gridaxes(grid, 1)) === 0:1.0:10
    @test @inferred(gridaxes(grid, 2)) === 1:2.0:20
    @test @inferred(gridorigin(grid))::Vec{2} == Vec(0,1)

    @test @inferred(Poingr.neighboring_nodes(grid, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 4:5))
    @test @inferred(Poingr.neighboring_nodes(grid, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 3:6))
    @test @inferred(Poingr.neighboring_cells(grid, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 3:5))
    @test @inferred(Poingr.neighboring_cells(grid, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 2:6))
    @test (Poingr.whichcell(grid, Vec(0.6, 8.8)))::CartesianIndex == CartesianIndex(1, 4)
    @test (Poingr.whichcell(grid, Vec(-0.6, 8.8)))::Nothing == nothing
end
