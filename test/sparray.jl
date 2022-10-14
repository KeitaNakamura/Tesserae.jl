using Marble: SpPattern, SpArray

@testset "SpPattern" begin
    @test @inferred(SpPattern((5,5))) == falses(5,5)
    @test @inferred(SpPattern(5,5)) == falses(5,5)

    spat = SpPattern(5,5)
    mask = rand(Bool,5,5)
    update_sparsity_pattern!(spat, mask)

    @test spat == mask
    for (i,I) in enumerate(findall(mask))
        @test spat.indices[I] == i
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
    A_spat = rand(Bool, size(A))
    B_spat = rand(Bool, size(B))

    for (x, x_spat) in ((A, A_spat), (B, B_spat))
        update_sparsity_pattern!(x, x_spat)
        @test x.spat == x_spat
        @test count(x.spat) == length(x.data)
        for i in eachindex(x)
            if x_spat[i]
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
    @test @inferred(A + A)::SpArray{Float64} == AA + AA
    @test @inferred(A + B)::SpArray{Float64} == map((x,y) -> ifelse(x==0||y==0,0,x+y), AA, BB)
    @test @inferred(A .* A)::SpArray{Float64} == AA .* AA
    @test @inferred(A .* B)::SpArray{Float64} == map((x,y) -> ifelse(x==0||y==0,0,x*y), AA, BB)
    @test @inferred(broadcast!(*, A, A, A))::SpArray{Float64} == broadcast!(*, AA, AA, AA)
    @test @inferred(broadcast!(*, A, A, B))::SpArray{Float64} == broadcast!(*, AA, AA, BB)
    @test A.spat == A_spat # sparsity pattern is never changed in `broadcast`
    @test @inferred(broadcast!(*, A, AA, B, 2))::SpArray{Float64} == broadcast!(*, AA, AA, BB, 2)
    @test A.spat == A_spat # sparsity pattern is never changed in `broadcast`
end
