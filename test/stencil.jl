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

    @testset "Region" begin
        physical = Physical()
        ghost⁻ = Ghost(-1)
        ghost⁺ = Ghost(+1)
        boundary⁻ = Boundary(-1)
        boundary⁺ = Boundary(+1)

        @test physical isa AxisRegion
        @test ghost⁻ isa AxisRegion
        @test boundary⁻ isa AxisRegion
        @test ghost⁻.side === -1
        @test ghost⁺.side === +1
        @test boundary⁻.side === -1
        @test boundary⁺.side === +1

        cells = @inferred Region(Cell(), physical; halo=2)
        @test cells isa Region{1}
        @test cells.placement == Cell()
        @test cells.axes == (physical,)
        @test cells.halo === 2

        lowghost = @inferred Region(Face(1), ghost⁻, physical; halo=1)
        @test lowghost isa Region{2}
        @test lowghost.placement == Face(1)
        @test lowghost.axes == (ghost⁻, physical)
        @test typeof(lowghost.axes) === Tuple{Ghost,Physical}
        @test lowghost.halo === 1
        @test isbitstype(typeof(lowghost))

        highboundary = @inferred Region(Face(1), boundary⁺, physical; halo=1)
        @test highboundary.axes == (boundary⁺, physical)
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
