using Tesserae.Stencil

@testset "Stencil" begin
    @testset "Placement" begin
        cell = @inferred Cell()
        face₁ = @inferred Face(1)
        face₂ = @inferred Face(2)
        face₃ = @inferred Face(3)

        @test typeof(cell) === typeof(face₁) === typeof(face₂) === typeof(face₃)
        @test isbitstype(typeof(cell))
        @test cell.mask === zero(UInt)
        @test face₁.mask === UInt(0b001)
        @test face₂.mask === UInt(0b010)
        @test face₃.mask === UInt(0b100)
    end

    @testset "GridOffset" begin
        offsets = @inferred unitoffsets(Val(3))
        e₁, e₂, e₃ = offsets

        @test length(offsets) == 3
        @test e₁.doubled == (2, 0, 0)
        @test e₂.doubled == (0, 2, 0)
        @test e₃.doubled == (0, 0, 2)
        @test typeof(e₁) === typeof(e₂) === typeof(e₃)
        @test isbitstype(typeof(e₁))

        @test +e₁ === e₁
        @test (@inferred(-e₁)).doubled == (-2, 0, 0)
        @test (@inferred(e₁ + e₂)).doubled == (2, 2, 0)
        @test (@inferred(e₁ - e₂)).doubled == (2, -2, 0)

        @test (@inferred(3e₁)).doubled == (6, 0, 0)
        @test (@inferred(e₁ * 3)).doubled == (6, 0, 0)
        @test (@inferred(e₁ / 2)).doubled == (1, 0, 0)
        @test (@inferred(3e₁ / 2)).doubled == (3, 0, 0)
        @test e₁ / 2 + e₁ / 2 == e₁

        z = @inferred zero(e₁)
        @test z == zero(typeof(e₁))
        @test z.doubled == (0, 0, 0)
        @test iszero(z)
        @test !iszero(e₁)

        @test sprint(show, e₁) == "GridOffset(1, 0, 0)"
        @test sprint(show, e₁ / 2) == "GridOffset(1//2, 0, 0)"

        @test_throws ArgumentError e₁ / 4
        @test_throws DivideError e₁ / 0
    end
end
