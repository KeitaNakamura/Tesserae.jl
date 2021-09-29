@testset "Index" begin
    li = LinearIndices((10,10))
    ci = CartesianIndices((10,10))
    index = Index(13, CartesianIndex(3,2))
    @test (@inferred Base.to_indices(li, (index,))) == (index.i,)
    @test (@inferred Base.to_indices(ci, (index,))) == Tuple(index.I)
end
