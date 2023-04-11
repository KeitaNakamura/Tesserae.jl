using Marble: SpIndices, SpArray

@testset "SpIndices" begin
    @test @inferred(SpIndices((5,5))) == zeros(5,5)
    @test @inferred(SpIndices(5,5)) == zeros(5,5)

    spinds = SpIndices(100,50)
    mask = rand(Bool, Marble.blocksize(spinds))
    Marble.blockindices(spinds) .= mask
    Marble.numbering!(spinds)

    inds = LinearIndices(Marble.nfill(1<<Marble.BLOCKFACTOR, Val(2)))
    offset = Ref(0)
    blks = map(mask) do isactive
        if isactive
            off = offset[]
            offset[] += length(inds)
            inds .+ off
        else
            zero(inds)
        end
    end
    @test spinds == hcat([vcat(blks[:,j]...) for j in 1:size(blks,2)]...)[axes(spinds)...]
end

@testset "SpArray" begin
    A = (@inferred SpArray{Float64}((100,50)))::SpArray{Float64, 2}
    A = (@inferred SpArray{Float64}(100,50))::SpArray{Float64, 2}

    @test all(==(0), A)

    B = SpArray{Int}(100,50)
    A_sppat = rand(Bool, Marble.blocksize(A))
    B_sppat = rand(Bool, Marble.blocksize(B))

    for (x, x_sppat) in ((A, A_sppat), (B, B_sppat))
        n = Marble.update_sparsity_pattern!(x, x_sppat)
        CI = CartesianIndices(x)
        @test n ≤ length(x.data)
        @test all(LinearIndices(x)) do i
            x[i] = i
            x[i] == ifelse(isnonzero(x, CI[i]), i, zero(eltype(x)))
        end
    end

    # broadcast
    AA = Array(A)
    BB = Array(B)
    @test @inferred(A + A)::Array{Float64} == AA + AA
    @test @inferred(A + B)::Array{Float64} == AA + BB
    @test @inferred(A .* A)::Array{Float64} == AA .* AA
    @test @inferred(A .* B)::Array{Float64} == AA .* BB
    @test @inferred(broadcast(iszero, A))::BitArray == @. iszero(AA)
    @test @inferred(broadcast!(*, A, A, A))::SpArray{Float64} == broadcast!(*, AA, AA, AA)
    @test_throws Exception broadcast!(*, A, A, B)
end
