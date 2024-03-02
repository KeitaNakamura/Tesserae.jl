@testset "Lattice" begin
    # constructor
    ## default
    (@inferred Lattice(Float32, 1, (0,3), (1,4)))::Lattice{2, Float32}
    (@inferred Lattice(Float64, 1, (0,3), (1,4), (0,2)))::Lattice{3, Float64}
    lattice = (@inferred Lattice(1, (0,3), (1,4), (0,2)))::Lattice{3, Float64}
    @test lattice[1] === Vec(0.0,1.0,0.0)
    @test lattice[end] === Vec(3.0,4.0,2.0)
    @test lattice == map(Vec, (Iterators.product(range(0,3,step=1), range(1,4,step=1), range(0,2,step=1))))
    ## from ranges
    lattice2 = (@inferred Lattice(range(0.0,3,step=1), range(1.0,4,step=1), range(0.0,2,step=1)))::Lattice{3, Float64}
    (@inferred Lattice(range(0.0f0,3,step=1), range(1.0f0,4,step=1), range(0.0f0,2,step=1)))::Lattice{3, Float32}
    @test lattice == lattice2
    @test_throws MethodError Lattice(range(0,3,step=1), range(1,4,step=1), range(0,2,step=1))

    # misc
    lattice = Lattice(0.2, (0,3), (0,4))
    @test size(lattice) === (16,21)
    @test IndexStyle(lattice) === IndexCartesian()
    @test (@inferred spacing(lattice)) === 0.2
    @test (@inferred spacing_inv(lattice)) === inv(0.2)

    # isinside
    @test (@inferred Sequoia.isinside(Vec(0.1,0.3), lattice)) === true
    @test (@inferred Sequoia.isinside(Vec(0.2,0.4), lattice)) === true
    ## exactly on the boundary
    @test (@inferred Sequoia.isinside(Vec(0.0,0.0), lattice)) === true
    @test (@inferred Sequoia.isinside(Vec(3.0,4.0), lattice)) === false
    ## outside
    @test (@inferred Sequoia.isinside(Vec(-1.0,3.0), lattice)) === false
    @test (@inferred Sequoia.isinside(Vec(1.0,-3.0), lattice)) === false

    # neighbornodes
    @test (@inferred neighbornodes(Vec(0.1,0.1), 1, lattice)) === CartesianIndices((1:2,1:2))
    @test (@inferred neighbornodes(Vec(0.3,0.1), 2, lattice)) === CartesianIndices((1:4,1:3))
    @test (@inferred neighbornodes(Vec(0.1,0.3), 2, lattice)) === CartesianIndices((1:3,1:4))
    ## exactly on the node
    @test (@inferred neighbornodes(Vec(0.2,0.4), 1, lattice)) === CartesianIndices((2:3,3:4))
    @test (@inferred neighbornodes(Vec(0.2,0.4), 2, lattice)) === CartesianIndices((1:4,2:5))
    @test (@inferred neighbornodes(Vec(3.0,4.0), 2, lattice)) === CartesianIndices((0:0,0:0))
    ## outside
    @test (@inferred neighbornodes(Vec(-0.1,3.05), 3, lattice)) === CartesianIndices((0:0,0:0))

    # whichcell
    @test Sequoia.whichcell(Vec(0.1,0.1), lattice) === CartesianIndex(1,1)
    @test Sequoia.whichcell(Vec(2.3,1.1), lattice) === CartesianIndex(12,6)
    ## exactly on the node
    @test Sequoia.whichcell(Vec(0.0,0.0), lattice) === CartesianIndex(1,1)
    @test Sequoia.whichcell(Vec(3.0,4.0), lattice) === nothing
end
