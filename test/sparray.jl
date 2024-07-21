@testset "SpIndex" begin
    @test Tesserae.isactive(Tesserae.SpIndex(1, 0)) === false
    @test Tesserae.isactive(Tesserae.SpIndex(1, 1)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(1, 2)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 0)) === false
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 1)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 2)) === true
    A = rand(3,3)
    @test (@inferred A[Tesserae.SpIndex(1,0)]) === zero(eltype(A))
    @test (@inferred A[Tesserae.SpIndex(1,1)]) === A[1]
    @test (@inferred A[Tesserae.SpIndex(CartesianIndex(1,2),0)]) === zero(eltype(A))
    @test (@inferred A[Tesserae.SpIndex(CartesianIndex(1,2),1)]) === A[1,2]
end

@testset "SpIndices" begin
    spinds = Tesserae.SpIndices(12,20)
    @test IndexStyle(spinds) === IndexCartesian()
    @test size(spinds) === (12,20)
    @test Tesserae.blocksize(spinds) === Tesserae.blocksize((12,20))
    @test !all(Tesserae.isactive, spinds)
    @test !all(i->Tesserae.isactive(spinds,i), eachindex(spinds))
    blkspy = rand(Bool, Tesserae.blocksize(spinds))
    n = update_block_sparsity!(spinds, blkspy)
    @test n == count(blkspy) * (2^Tesserae.BLOCKFACTOR)^2 # `^2` is for dimension
    @test n == Tesserae.countnnz(spinds)
    inds = zeros(Int, size(spinds))
    for I in CartesianIndices(inds)
        blk, i = Tesserae.blocklocal(Tuple(I)...)
        if blkspy[blk...]
            linear_blkindex = LinearIndices(blkspy)[blk...]
            nblks = count(blkspy[1:linear_blkindex])
            blkunit = (2^Tesserae.BLOCKFACTOR)^2
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
    @test !all(i->Tesserae.isactive(A,i), eachindex(A))
    @test all(iszero, A)
    blkspy = rand(Bool, Tesserae.blocksize(A))
    n = update_block_sparsity!(A, blkspy)
    @test length(Tesserae.get_data(A)) === n
    @test all(i->(A[i]=rand(); iszero(A[i])), filter(i->!Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->(a=rand(); A[i]=a; A[i]==a), filter(i->Tesserae.isactive(A,i), eachindex(A)))

    # breadcast is tested on grid part
end
