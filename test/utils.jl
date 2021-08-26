@testset "Index" begin
    li = LinearIndices((10,10))
    ci = CartesianIndices((10,10))
    @test (@inferred Index(li, 13))::Index{2} === Index(13, CartesianIndex(3,2))
    @test (@inferred Index(ci, 13))::Index{2} === Index(13, CartesianIndex(3,2))
    @test (@inferred Index(li, CartesianIndex(3,2)))::Index{2} === Index(13, CartesianIndex(3,2))
    @test (@inferred Index(ci, CartesianIndex(3,2)))::Index{2} === Index(13, CartesianIndex(3,2))
    index = Index(13, CartesianIndex(3,2))
    @test (@inferred Base.to_indices(li, (index,))) == (index.i,)
    @test (@inferred Base.to_indices(ci, (index,))) == Tuple(index.I)
end
