@testset "Grid" begin
    # constructors
    @test @inferred(Grid(0:10))::Grid{1, Int} == Vec.(0:10)
    @test @inferred(Grid(0:10, 0:20))::Grid{2, Int} == Vec.(collect(Iterators.product(0:10, 0:20)))
    @test @inferred(Grid(0:10, 0:20, 0:30))::Grid{3, Int} == Vec.(collect(Iterators.product(0:10, 0:20, 0:30)))
    @test @inferred(Grid{1}(0:10))::Grid{1, Int} == Vec.(0:10)
    @test @inferred(Grid{2}(0:10))::Grid{2, Int} == Vec.(collect(Iterators.product(0:10, 0:10)))
    @test @inferred(Grid{3}(0:10))::Grid{3, Int} == Vec.(collect(Iterators.product(0:10, 0:10, 0:10)))

    # gridsteps/gridaxes/gridorigin
    grid = Grid(0:1.0:10, 1:2.0:20)
    @test @inferred(gridsteps(grid)) === (1.0, 2.0)
    @test @inferred(gridsteps(grid, 1)) === 1.0
    @test @inferred(gridsteps(grid, 2)) === 2.0
    @test @inferred(gridaxes(grid)) === (0:1.0:10, 1:2.0:20)
    @test @inferred(gridaxes(grid, 1)) === 0:1.0:10
    @test @inferred(gridaxes(grid, 2)) === 1:2.0:20
    @test @inferred(gridorigin(grid))::Vec{2} == Vec(0,1)

    # neighboring_nodes/neighboring_cells/whichcell/whichblock
    @test @inferred(Poingr.neighboring_nodes(grid, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 4:5))
    @test @inferred(Poingr.neighboring_nodes(grid, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 3:6))
    @test @inferred(Poingr.neighboring_cells(grid, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 3:5))
    @test @inferred(Poingr.neighboring_cells(grid, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 2:6))
    @test_throws BoundsError Poingr.neighboring_cells(grid, CartesianIndex(11, 9), 1)
    @test @inferred(Poingr.neighboring_blocks(grid, Vec(8.8, 4.6), 1))::CartesianIndices == CartesianIndices((1:2, 1:2))
    @test @inferred(Poingr.neighboring_blocks(grid, Vec(8.8, 4.6), 2))::CartesianIndices == CartesianIndices((1:2, 1:2))
    @test_throws BoundsError Poingr.neighboring_blocks(grid, CartesianIndex(3, 1), 1)
    @test (Poingr.whichcell(grid, Vec(0.6, 8.8)))::CartesianIndex == CartesianIndex(1, 4)
    @test (Poingr.whichcell(grid, Vec(-0.6, 8.8)))::Nothing == nothing
    @test (Poingr.whichblock(grid, Vec(8.8, 4.6)))::CartesianIndex == CartesianIndex(2, 1)
    @test (Poingr.whichblock(grid, Vec(-8.8, 4.6)))::Nothing == nothing

    # pointsinblock
    @test Poingr.blocksize(grid) == (2, 2)
    xₚ = Vec{2, Float64}[(2,2), (8, 18), (8, 21), (4, 18), (5, 18)]
    @test Poingr.pointsinblock(grid, xₚ) == reshape([[1], [], [4, 5], [2]], 2, 2)

    # coloringblocks
    @test Poingr.coloringblocks((20, 30)) == [[CartesianIndex(1,1) CartesianIndex(1,3); CartesianIndex(3,1) CartesianIndex(3,3)],
                                              [CartesianIndex(2,1) CartesianIndex(2,3)],
                                              [CartesianIndex(1,2) CartesianIndex(1,4); CartesianIndex(3,2) CartesianIndex(3,4)],
                                              [CartesianIndex(2,2) CartesianIndex(2,4)]]
end
