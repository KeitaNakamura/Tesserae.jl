@testset "Collection" begin
    c = Collection([1,2,3])
    @test Collections.whichrank(c) == 1
    for i in eachindex(c)
        @test (@inferred c[i])::Int == i
    end
    for i in eachindex(c)
        c[i] = 2i
        @test (@inferred c[i])::Int == 2i
    end
end
