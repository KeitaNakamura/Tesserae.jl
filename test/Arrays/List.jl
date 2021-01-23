@testset "List/ListGroup" begin
    struct Item <: ListGroup{Item}
        a::Int
    end
    for (ItemName, GroupName) in ((:MyType, :MyType), # with type
                                  (:MyType2, Meta.quot(:MyType))) # with symbol
        @eval begin
            struct $ItemName <: ListGroup{$GroupName}
                a::Int
            end
            x = (@inferred $ItemName(3) + $ItemName(2) + $ItemName(1))::List{$GroupName, 3}
            @test Tuple(x) == ($ItemName(3), $ItemName(2), $ItemName(1))
            @test x[1] == $ItemName(3)
            @test x[2] == $ItemName(2)
            @test x[3] == $ItemName(1)
            @test length(x) == 3
            @test eachindex(x) == Base.OneTo(3)
            @test collect(x) == collect(Tuple(x))

            @test_throws Exception $ItemName(1) + Item(1)
            @test_throws Exception Item(1) + $ItemName(1)
            @test_throws Exception x + Item(1)
            @test_throws Exception Item(1) + x
        end
    end
end
