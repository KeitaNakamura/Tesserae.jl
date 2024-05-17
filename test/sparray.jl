@testset "SpIndex" begin
    @test Sequoia.isactive(Sequoia.SpIndex(1, 0)) === false
    @test Sequoia.isactive(Sequoia.SpIndex(1, 1)) === true
    @test Sequoia.isactive(Sequoia.SpIndex(1, 2)) === true
    @test Sequoia.isactive(Sequoia.SpIndex(CartesianIndex(1,1), 0)) === false
    @test Sequoia.isactive(Sequoia.SpIndex(CartesianIndex(1,1), 1)) === true
    @test Sequoia.isactive(Sequoia.SpIndex(CartesianIndex(1,1), 2)) === true
    A = rand(3,3)
    @test (@inferred A[Sequoia.SpIndex(1,0)]) === zero(eltype(A))
    @test (@inferred A[Sequoia.SpIndex(1,1)]) === A[1]
    @test (@inferred A[Sequoia.SpIndex(CartesianIndex(1,2),0)]) === zero(eltype(A))
    @test (@inferred A[Sequoia.SpIndex(CartesianIndex(1,2),1)]) === A[1,2]
end

@testset "SpIndices" begin
    spinds = Sequoia.SpIndices(12,20)
    @test IndexStyle(spinds) === IndexCartesian()
    @test size(spinds) === (12,20)
    @test Sequoia.blocksize(spinds) === Sequoia.blocksize((12,20))
    @test !all(Sequoia.isactive, spinds)
    @test !all(i->Sequoia.isactive(spinds,i), eachindex(spinds))
    blkspy = rand(Bool, Sequoia.blocksize(spinds))
    n = update_block_sparsity!(spinds, blkspy)
    @test n == count(blkspy) * (2^Sequoia.BLOCKFACTOR)^2 # `^2` is for dimension
    @test n == Sequoia.countnnz(spinds)
    inds = zeros(Int, size(spinds))
    for I in CartesianIndices(inds)
        blk, i = Sequoia.blocklocal(Tuple(I)...)
        if blkspy[blk...]
            linear_blkindex = LinearIndices(blkspy)[blk...]
            nblks = count(blkspy[1:linear_blkindex])
            blkunit = (2^Sequoia.BLOCKFACTOR)^2
            index = (nblks-1) * blkunit + i
            inds[I] = index
        end
    end
    @test map(i->i.spindex, spinds) == inds
end

@testset "SpArray" begin
    A = SpArray{Float64}(undef, 12, 20)
    @test IndexStyle(A) === IndexCartesian()
    @test size(A) === (12,20)
    @test !all(i->Sequoia.isactive(A,i), eachindex(A))
    @test all(iszero, A)
    blkspy = rand(Bool, Sequoia.blocksize(A))
    n = update_block_sparsity!(A, blkspy)
    @test length(Sequoia.get_data(A)) === n
    @test all(i->(A[i]=rand(); iszero(A[i])), filter(i->!Sequoia.isactive(A,i), eachindex(A)))
    @test all(i->(a=rand(); A[i]=a; A[i]==a), filter(i->Sequoia.isactive(A,i), eachindex(A)))

    # breadcast is tested on grid part
end
