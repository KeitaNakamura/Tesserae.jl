@testset "FillArray" begin
    val = 2
    arr = fill(val, 2, 3)
    for T in (Float32, Float64)
        @test (@inferred FillArray{T}(val, (2, 3)))::FillArray{T, 2} == arr
        @test (@inferred FillArray{T}(val, 2, 3))::FillArray{T, 2} == arr
        @test (@inferred FillArray(val, 2, 3))::FillArray{Int, 2} == arr
    end
end

@testset "Ones/Zeros" begin
    for (FillType, val) in ((Ones, 1), (Zeros, 0))
        arr = fill(val, 2, 3)
        for T in (Float32, Float64, Int)
            @test (@inferred FillType{T}((2, 3)))::FillType{T, 2} == arr
            @test (@inferred FillType{T}(2, 3))::FillType{T, 2} == arr
            @test (@inferred FillType((2, 3)))::FillType{Float64, 2} == arr
            @test (@inferred FillType(2, 3))::FillType{Float64, 2} == arr
        end
    end
end
