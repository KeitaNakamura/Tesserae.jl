@testset "AbstractCollection" begin
    struct MyCollection{T} <: AbstractCollection{1}
        data::Vector{T}
    end
    Base.getindex(x::MyCollection, i::Int) = x.data[i]
    Base.setindex!(x::MyCollection, v, i::Int) = x.data[i] = v
    Base.length(x::MyCollection) = length(x.data)

    data = [1,2,3]
    for T in (Float32, Float64)
        x = MyCollection{T}(data)
        # eltype
        @test (@inferred eltype(x)) == T
        # checkbounds(Bool, ...)
        @test (@inferred checkbounds(Bool, x, 1))::Bool == true
        @test (@inferred checkbounds(Bool, x, 2))::Bool == true
        @test (@inferred checkbounds(Bool, x, 3))::Bool == true
        @test (@inferred checkbounds(Bool, x, -1))::Bool == false
        @test (@inferred checkbounds(Bool, x, 4))::Bool == false
        # checkbounds(...)
        @test (@inferred checkbounds(x, 1)) === nothing
        @test (@inferred checkbounds(x, 2)) === nothing
        @test (@inferred checkbounds(x, 3)) === nothing
        @test_throws Exception checkbounds(x, -1)
        @test_throws Exception checkbounds(x, 4)
        # size/eachindex/firstindex/lastindex
        @test size(x) == (3,)
        @test eachindex(x) == Base.OneTo(3)
        @test firstindex(x) == 1
        @test lastindex(x) == 3
        # iterate/Array
        @test collect(x) == [1,2,3]
        @test Array(x) == [1,2,3]
        # fill!
        fill!(x, 10)
        @test (@inferred x[1])::T == 10
        @test (@inferred x[2])::T == 10
        @test (@inferred x[3])::T == 10
    end
end
