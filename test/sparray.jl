@testset "SpPattern" begin
    @test @inferred(Poingr.SpPattern((5,5))) == falses(5,5)
    @test @inferred(Poingr.SpPattern(5,5)) == falses(5,5)

    spat = Poingr.SpPattern(5,5)

    # getindex/setindex!
    inds = fill(-1, size(spat))
    true_inds = 1:2:length(spat)
    for (i, I) in enumerate(true_inds)
        spat[I] = true
        @test spat[I] == true
        inds[I] = i # for latter test
    end
    for i in setdiff(1:length(spat), true_inds)
        @test spat[i] == false
    end

    # reinit!
    @test Poingr.reinit!(spat) == count(spat)
    @test spat.indices == inds

    # broadcast
    mask2 = Poingr.SpPattern(size(spat))
    mask2 .= rand(Bool, size(spat))
    @test @inferred(spat .| mask2)::Poingr.SpPattern == Array(spat) .| Array(mask2)
    @test @inferred(spat .& mask2)::Poingr.SpPattern == Array(spat) .& Array(mask2)
    @test @inferred(spat .| Array(mask2))::Poingr.SpPattern == Array(spat) .| Array(mask2)
    @test @inferred(spat .& Array(mask2))::Poingr.SpPattern == Array(spat) .& Array(mask2)

    # fill!
    fill!(spat, false)
    @test all(==(false), spat)
end

@testset "SpArray" begin
    A = (@inferred Poingr.SpArray{Float64}(undef,(5,5)))::Poingr.SpArray{Float64, 2, Vector{Float64}}
    A = (@inferred Poingr.SpArray{Float64}(undef,5,5))::Poingr.SpArray{Float64, 2, Vector{Float64}}

    @test all(==(0), A)
    for i in eachindex(A)
        # @test_throws Exception A[i] = 1
    end

    B = Poingr.SpArray{Int}(undef,5,5)
    A_spat = rand(Bool, size(A))
    B_spat = rand(Bool, size(B))

    for (x, x_spat) in ((A, A_spat), (B, B_spat))
        x.spat .= x_spat
        Poingr.reinit!(x)
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
    @test @inferred(A + A)::Poingr.SpArray{Float64} == AA + AA
    @test @inferred(A + B)::Poingr.SpArray{Float64} == AA + BB
    @test @inferred(A .* A)::Poingr.SpArray{Float64} == AA .* AA
    @test @inferred(A .* B)::Poingr.SpArray{Float64} == AA .* BB
    @test @inferred(broadcast!(*, A, A, A))::Poingr.SpArray{Float64} == broadcast!(*, AA, AA, AA)
    @test @inferred(broadcast!(*, A, A, B))::Poingr.SpArray{Float64} == broadcast!(*, AA, AA, BB)
    @test A.spat == A_spat # sparsity pattern is never changed in `broadcast`
    @test @inferred(broadcast!(*, A, AA, B, 2))::Poingr.SpArray{Float64} == broadcast!(*, AA, AA, BB, 2)
    @test A.spat == A_spat # sparsity pattern is never changed in `broadcast`
end
