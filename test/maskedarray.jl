@testset "Mask" begin
    @test @inferred(Poingr.Mask((5,5))) == falses(5,5)
    @test @inferred(Poingr.Mask(5,5)) == falses(5,5)

    mask = Poingr.Mask(5,5)

    # getindex/setindex!
    inds = fill(-1, size(mask))
    true_inds = 1:2:length(mask)
    for (i, I) in enumerate(true_inds)
        mask[I] = true
        @test mask[I] == true
        inds[I] = i # for latter test
    end
    for i in setdiff(1:length(mask), true_inds)
        @test mask[i] == false
    end

    # reinit!
    @test reinit!(mask) == count(mask)
    @test mask.indices == inds

    # broadcast
    mask2 = Poingr.Mask(size(mask))
    mask2 .= rand(Bool, size(mask))
    @test @inferred(mask .| mask2)::Poingr.Mask == Array(mask) .| Array(mask2)
    @test @inferred(mask .& mask2)::Poingr.Mask == Array(mask) .& Array(mask2)
    @test @inferred(mask .| Array(mask2))::Poingr.Mask == Array(mask) .| Array(mask2)
    @test @inferred(mask .& Array(mask2))::Poingr.Mask == Array(mask) .& Array(mask2)

    # fill!
    fill!(mask, false)
    @test all(==(false), mask)
end

@testset "MaskedArray" begin
    A = (@inferred Poingr.MaskedArray{Float64}(undef,(5,5)))::Poingr.MaskedArray{Float64, 2, Vector{Float64}}
    A = (@inferred Poingr.MaskedArray{Float64}(undef,5,5))::Poingr.MaskedArray{Float64, 2, Vector{Float64}}

    @test all(==(0), A)
    for i in eachindex(A)
        @test_throws Exception A[i] = 1
    end

    B = Poingr.MaskedArray{Int}(undef,5,5)
    A_mask = rand(Bool, size(A))
    B_mask = rand(Bool, size(B))

    for (x, x_mask) in ((A, A_mask), (B, B_mask))
        x.mask .= x_mask
        reinit!(x)
        @test x.mask == x_mask
        @test count(x.mask) == length(x.data)
        for i in eachindex(x)
            if x_mask[i]
                x[i] = i
                @test x[i] == i
            else
                @test_throws Exception x[i] = i
            end
        end
    end

    # broadcast
    AA = Array(A)
    BB = Array(B)
    @test @inferred(A + A)::Poingr.MaskedArray{Float64} == AA + AA
    @test @inferred(A + B)::Poingr.MaskedArray{Float64} == AA + BB
    @test @inferred(A .* A)::Poingr.MaskedArray{Float64} == AA .* AA
    @test @inferred(A .* B)::Poingr.MaskedArray{Float64} == AA .* BB
    @test @inferred(broadcast!(*, A, A, A))::Poingr.MaskedArray{Float64} == broadcast!(*, AA, AA, AA)
    @test @inferred(broadcast!(*, A, A, B))::Poingr.MaskedArray{Float64} == broadcast!(*, AA, AA, BB)
    @test A.mask == A_mask .| B_mask
    @test @inferred(broadcast!(*, A, AA, B, 2))::Poingr.MaskedArray{Float64} == broadcast!(*, AA, AA, BB, 2)
    @test all(A.mask)
end
