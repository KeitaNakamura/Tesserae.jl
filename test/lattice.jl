@testset "Lattice" begin
    # constructors
    for T in (Float32, Float64)
        dx = 2.0
        for minmax in (((0,10),), ((0,10), (0,20)), ((0,10), (0,20), (0,30)))
            axes = map(x->range(x...;step=dx), minmax)
            @test (@inferred Lattice(dx, minmax...))::Lattice{length(axes), Float64} == Vec.(collect(Iterators.product(axes...)))
            @test (@inferred Lattice(T, dx, minmax...))::Lattice{length(axes), T} == Vec.(collect(Iterators.product(axes...)))
        end
    end

    # spacing/get_axes
    lattice = Lattice(1.0, (0,10), (1,20))
    @test @inferred(spacing(lattice)) == 1.0
    @test @inferred(Marble.get_axes(lattice)) == (0:1.0:10, 1:1.0:20)
    @test @inferred(Marble.get_axes(lattice, 1)) == 0:1.0:10
    @test @inferred(Marble.get_axes(lattice, 2)) == 1:1.0:20

    # getindex
    @test lattice[2:8, 5:9] == Array(lattice[2:8, 5:9])

    # isinside
    @test Marble.isinside(Vec(0,1), lattice) == true
    @test Marble.isinside(Vec(2,4), lattice) == true
    @test Marble.isinside(Vec(0,0), lattice) == false
    @test_throws Exception Marble.isinside(Vec(1), lattice)

    # neighbornodes/whichcell/whichblock
    @test @inferred(neighbornodes(lattice, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 8:9))
    @test @inferred(neighbornodes(lattice, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 7:10))
    @test @inferred(neighbornodes(lattice, Vec(-0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:0, 1:0))
    @test (Marble.whichcell(lattice, Vec(0.6, 8.8)))::CartesianIndex == CartesianIndex(1, 8)
    @test (Marble.whichcell(lattice, Vec(-0.6, 8.8)))::Nothing == nothing
    @test (Marble.whichblock(lattice, Vec(8.8, 4.6)))::CartesianIndex == CartesianIndex(2, 1)
    @test (Marble.whichblock(lattice, Vec(-8.8, 4.6)))::Nothing == nothing

    # pointsperblock
    @test Marble.blocksize(size(lattice)) == (2, 3)
    xₚ = Vec{2, Float64}[(2,2), (8.5, 18), (8.5, 21), (4.3, 18), (5, 14)]
    @test Marble.pointsperblock(lattice, xₚ) == reshape([[1], [5], [4],
                                                         [ ], [ ], [2]], 3,2) |> permutedims

    # threadsafe_blocks
    @test Marble.threadsafe_blocks(Marble.blocksize((20, 30))) == [[CartesianIndex(1,1) CartesianIndex(1,3); CartesianIndex(3,1) CartesianIndex(3,3)],
                                                                   [CartesianIndex(2,1) CartesianIndex(2,3)],
                                                                   [CartesianIndex(1,2) CartesianIndex(1,4); CartesianIndex(3,2) CartesianIndex(3,4)],
                                                                   [CartesianIndex(2,2) CartesianIndex(2,4)]]
end
