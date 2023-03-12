@testset "BlockSpace" begin
    for T in (Float32, Float64)
        lattice = Lattice(T, 1.0, (0,10), (1,20))
        # BlockSpace
        @test Marble.blocksize(size(lattice)) == (2, 3)
        xₚ = Vec{2,T}[(2,2), (8.5, 18), (8.5, 21), (4.3, 18), (5, 14)]
        blkspace = Marble.BlockSpace(Marble.blocksize(size(lattice)), length(xₚ))
        update!(blkspace, lattice, xₚ)
        @test Marble.particlesinblocks(blkspace) == reshape([[1], [5], [4],
                                                             [ ], [ ], [2]], 3,2) |> permutedims
    end
    # threadsafe_blocks
    @test Marble.threadsafe_blocks(Marble.blocksize((20, 30))) == [[CartesianIndex(1,1) CartesianIndex(1,3); CartesianIndex(3,1) CartesianIndex(3,3)],
                                                                   [CartesianIndex(2,1) CartesianIndex(2,3)],
                                                                   [CartesianIndex(1,2) CartesianIndex(1,4); CartesianIndex(3,2) CartesianIndex(3,4)],
                                                                   [CartesianIndex(2,2) CartesianIndex(2,4)]]
end
