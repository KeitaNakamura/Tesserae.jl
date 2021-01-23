@testset "ScalarMatrix" begin
    I = [1 0 0; 0 1 0; 0 0 1]
    for T in (Float32, Float64)
        @test (@inferred ScalarMatrix{T}(2, 3))::ScalarMatrix{T} == 2 * I
        @test (@inferred ScalarMatrix(2, 3))::ScalarMatrix{Int} == 2 * I
    end
end
