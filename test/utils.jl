@testset "Utilities" begin
    # nfill
    @test (@inferred Tesserae.nfill(3, Val(3)))   === (3,3,3)
    @test (@inferred Tesserae.nfill(1.2, Val(2))) === (1.2,1.2)
    # zero_recursive
    @test (@inferred Tesserae.zero_recursive(1)) === 0
    @test (@inferred Tesserae.zero_recursive(1.2)) === 0.0
    @test (@inferred Tesserae.zero_recursive(1.2f0)) === 0.0f0
    @test (@inferred Tesserae.zero_recursive((;x=1,y=2.0,z=3.0f0))) === (;x=0,y=0.0,z=0.0f0)
    @test (@inferred Tesserae.zero_recursive((;x=1,y=2.0,z=(;a=3.0f0)))) === (;x=0,y=0.0,z=(;a=0.0f0))
    @test (@inferred Tesserae.zero_recursive(Int)) === 0
    @test (@inferred Tesserae.zero_recursive(Float64)) === 0.0
    @test (@inferred Tesserae.zero_recursive(Float32)) === 0.0f0
    @test (@inferred Tesserae.zero_recursive(@NamedTuple{x::Int,y::Float64,z::Float32})) === (;x=0,y=0.0,z=0.0f0)
    @test (@inferred Tesserae.zero_recursive(@NamedTuple{x::Int,y::Float64,z::@NamedTuple{a::Float32}})) === (;x=0,y=0.0,z=(;a=0.0f0))
    @test_throws ArgumentError Tesserae.zero_recursive(zeros(3))
    @test_throws ArgumentError Tesserae.zero_recursive("abc")
    # fillzero!
    @test (@inferred Tesserae.fillzero!(rand(3)))::Vector{Float64} == zeros(3)
    # commas
    @test (@inferred Tesserae.commas(123))::String == "123"
    @test (@inferred Tesserae.commas(1234))::String == "1,234"
    @test (@inferred Tesserae.commas(12345678))::String == "12,345,678"
    # maparray
    @test (@inferred Tesserae.maparray(Float32, [1,2,3,4])) == Float32[1,2,3,4]
    @test (@inferred Tesserae.maparray(Float64, [1,2,3,4])) == Float64[1,2,3,4]
    @test (@inferred Tesserae.maparray(sqrt, [1,2,3,4])) == map(sqrt, [1,2,3,4])
    # Trues
    @test (@inferred Tesserae.Trues((3,2))) == trues(3,2)
end
