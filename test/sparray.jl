using Marble: SpPattern, SpArray

@testset "SpPattern" begin
    @test @inferred(SpPattern((5,5))) == falses(5,5)
    @test @inferred(SpPattern(5,5)) == falses(5,5)

    sppat = SpPattern(5,5)
    mask = rand(Bool,5,5)
    update_sparsity_pattern!(sppat, mask)

    @test sppat == mask
    for (i,I) in enumerate(findall(mask))
        @test sppat.indices[I] == i
    end
end

@testset "SpArray" begin
    A = (@inferred SpArray{Float64}((5,5)))::SpArray{Float64, 2, Vector{Float64}}
    A = (@inferred SpArray{Float64}(5,5))::SpArray{Float64, 2, Vector{Float64}}

    @test all(==(0), A)
    for i in eachindex(A)
        # @test_throws Exception A[i] = 1
    end

    B = SpArray{Int}(5,5)
    A_sppat = rand(Bool, size(A))
    B_sppat = rand(Bool, size(B))

    for (x, x_sppat) in ((A, A_sppat), (B, B_sppat))
        update_sparsity_pattern!(x, x_sppat)
        @test x.sppat == x_sppat
        @test count(x.sppat) == length(x.data)
        for i in eachindex(x)
            if x_sppat[i]
                x[i] = i
                @test x[i] == i
            else
                # @test_throws Exception x[i] = i
            end
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
