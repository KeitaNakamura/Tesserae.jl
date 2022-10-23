@testset "Grid" begin
    # constructors
    for T in (Float32, Float64)
        for axs in ((0:10,), (0:10, 0:20), (0:10, 0:20, 0:30))
            @test (@inferred Grid(axs...))::Grid{Float64, length(axs)} == Vec.(collect(Iterators.product(axs...)))
            @test (@inferred Grid{T}(axs...))::Grid{T, length(axs)} == Vec.(collect(Iterators.product(axs...)))
        end
    end

    # gridsteps/gridaxes/gridorigin
    grid = Grid(0:1.0:10, 1:2.0:20)
    @test @inferred(gridsteps(grid)) == (1.0, 2.0)
    @test @inferred(gridsteps(grid, 1)) == 1.0
    @test @inferred(gridsteps(grid, 2)) == 2.0
    @test @inferred(gridaxes(grid)) == (0:1.0:10, 1:2.0:20)
    @test @inferred(gridaxes(grid, 1)) == 0:1.0:10
    @test @inferred(gridaxes(grid, 2)) == 1:2.0:20
    @test @inferred(gridorigin(grid))::Vec{2} == Vec(0,1)

    # isinside
    @test Marble.isinside(Vec(0,1), grid) == true
    @test Marble.isinside(Vec(2,4), grid) == true
    @test Marble.isinside(Vec(0,0), grid) == false
    @test_throws Exception Marble.isinside(Vec(1), grid)

    # nodeindices/whichcell/whichblock
    @test @inferred(Marble.nodeindices(grid, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 4:5))
    @test @inferred(Marble.nodeindices(grid, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 3:6))
    @test @inferred(Marble.nodeindices(grid, Vec(-0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:0, 1:0))
    @test @inferred(Marble.nodeindices(grid, Vec(0.6, 8.8), Vec(1,2)))::CartesianIndices == CartesianIndices((1:2, 3:6))
    @test (Marble.whichcell(grid, Vec(0.6, 8.8)))::CartesianIndex == CartesianIndex(1, 4)
    @test (Marble.whichcell(grid, Vec(-0.6, 8.8)))::Nothing == nothing
    @test (Marble.whichblock(grid, Vec(8.8, 4.6)))::CartesianIndex == CartesianIndex(2, 1)
    @test (Marble.whichblock(grid, Vec(-8.8, 4.6)))::Nothing == nothing

    # pointsperblock
    @test Marble.blocksize(size(grid)) == (2, 2)
    xₚ = Vec{2, Float64}[(2,2), (8, 18), (8, 21), (4, 18), (5, 18)]
    @test Marble.pointsperblock(grid, xₚ) == reshape([[1], [], [4, 5], [2]], 2, 2)

    # threadsafe_blocks
    @test Marble.threadsafe_blocks(Marble.blocksize((20, 30))) == [[CartesianIndex(1,1) CartesianIndex(1,3); CartesianIndex(3,1) CartesianIndex(3,3)],
                                                                   [CartesianIndex(2,1) CartesianIndex(2,3)],
                                                                   [CartesianIndex(1,2) CartesianIndex(1,4); CartesianIndex(3,2) CartesianIndex(3,4)],
                                                                   [CartesianIndex(2,2) CartesianIndex(2,4)]]
end
