@testset "Lattice" begin
    for T in (Float32, Float64)
        # constructors
        dx = 2.0
        for minmax in (((0,10),), ((0,10), (0,20)), ((0,10), (0,20), (0,30)))
            axes = map(x->range(x...;step=dx), minmax)
            @test (@inferred Lattice(dx, minmax...))::Lattice{length(axes), Float64} == Vec.(collect(Iterators.product(axes...)))
            @test (@inferred Lattice(T, dx, minmax...))::Lattice{length(axes), T} == Vec.(collect(Iterators.product(axes...)))
        end

        # spacing/get_axes
        lattice = Lattice(T, 1.0, (0,10), (1,20))
        @test @inferred(spacing(lattice)) == 1.0
        @test @inferred(Marble.get_axes(lattice)) == (0:1.0:10, 1:1.0:20)
        @test @inferred(Marble.get_axes(lattice, 1)) == 0:1.0:10
        @test @inferred(Marble.get_axes(lattice, 2)) == 1:1.0:20

        # getindex
        @test lattice[2:8, 5:9] == Array(lattice[2:8, 5:9])

        # isinside
        @test Marble.isinside(Vec{2,T}(0,1), lattice) == true
        @test Marble.isinside(Vec{2,T}(2,4), lattice) == true
        @test Marble.isinside(Vec{2,T}(0,0), lattice) == false
        @test_throws Exception Marble.isinside(Vec(1), lattice)

        # neighbornodes/whichcell/whichblock
        @test @inferred(neighbornodes(lattice, Vec{2,T}( 1.6,8.8), 1))::Tuple{CartesianIndices, Bool} == (CartesianIndices((2:3, 8:9)), true)
        @test @inferred(neighbornodes(lattice, Vec{2,T}( 0.6,8.8), 2))::Tuple{CartesianIndices, Bool} == (CartesianIndices((1:3, 7:10)), false)
        @test @inferred(neighbornodes(lattice, Vec{2,T}(-0.6,8.8), 2))::Tuple{CartesianIndices, Bool} == (CartesianIndices((1:0, 1:0)), false)
        @test (Marble.whichcell(lattice, Vec{2,T}( 0.6,8.8)))::CartesianIndex == CartesianIndex(1, 8)
        @test (Marble.whichcell(lattice, Vec{2,T}(-0.6,8.8)))::Nothing == nothing
        @test (Marble.whichblock(lattice, Vec{2,T}( 8.8,4.6)))::CartesianIndex == CartesianIndex(2, 1)
        @test (Marble.whichblock(lattice, Vec{2,T}(-8.8,4.6)))::Nothing == nothing

        # ParticlesInBlocks
        @test Marble.blocksize(size(lattice)) == (2, 3)
        xₚ = Vec{2,T}[(2,2), (8.5, 18), (8.5, 21), (4.3, 18), (5, 14)]
        ptsinblks = Marble.ParticlesInBlocks(Marble.blocksize(size(lattice)), length(xₚ))
        Marble.update_sparsity_pattern!(ptsinblks, lattice, xₚ)
        @test ptsinblks == reshape([[1], [5], [4],
                                    [ ], [ ], [2]], 3,2) |> permutedims
    end

    # threadsafe_blocks
    @test Marble.threadsafe_blocks(Marble.blocksize((20, 30))) == [[CartesianIndex(1,1) CartesianIndex(1,3); CartesianIndex(3,1) CartesianIndex(3,3)],
                                                                   [CartesianIndex(2,1) CartesianIndex(2,3)],
                                                                   [CartesianIndex(1,2) CartesianIndex(1,4); CartesianIndex(3,2) CartesianIndex(3,4)],
                                                                   [CartesianIndex(2,2) CartesianIndex(2,4)]]
end
