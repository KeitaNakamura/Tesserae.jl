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

        e, = unitoffsets(Val(1))
        @test iszero(cells.offset)

        shifted = @inferred cells + e / 2
        @test shifted.placement == cells.placement
        @test shifted.axes == cells.axes
        @test shifted.halo == cells.halo
        @test shifted.offset == e / 2
        @test (@inferred(e / 2 + cells)) == shifted
        @test (@inferred(shifted - e)).offset == -e / 2
        @test (@inferred(shifted + e / 2)).offset == e
        @test_throws ArgumentError e - cells

        lowghost = @inferred Region(Face(1), ghost⁻, physical; halo=1)
        @test lowghost isa Region{2}
        @test lowghost.placement == Face(1)
        @test lowghost.axes == (ghost⁻, physical)
        @test typeof(lowghost.axes) === Tuple{Ghost,Physical}
        @test lowghost.halo === 1
        @test isbitstype(typeof(lowghost))

        e₁, e₂ = unitoffsets(Val(2))
        translated = @inferred lowghost + e₁ - e₂ / 2
        @test translated.offset.doubled == (2, -1)

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

    @testset "Index ranges" begin
        e, = unitoffsets(Val(1))
        cells = Region(Cell(), Physical(); halo=2)
        faces = Region(Face(1), Physical(); halo=2)
        cell_axes = axes(zeros(10))
        face_axes = axes(zeros(11))

        @test (@inferred Stencil.indexranges(cells, cell_axes)) == (3:8,)
        @test (@inferred Stencil.indexranges(faces, face_axes)) == (3:9,)

        @test Stencil.indexranges(cells + e / 2, face_axes) == (4:9,)
        @test Stencil.indexranges(cells - e / 2, face_axes) == (3:8,)
        @test Stencil.indexranges(faces + e / 2, cell_axes) == (3:9,)
        @test Stencil.indexranges(faces - e / 2, cell_axes) == (2:8,)

        @test Stencil.indexranges(cells + e, cell_axes) == (4:9,)
        @test Stencil.indexranges(cells - e, cell_axes) == (2:7,)

        e₁, e₂ = unitoffsets(Val(2))
        region = Region(Face(1), Physical(), Physical(); halo=1) + (e₁ - e₂) / 2
        @test Stencil.indexranges(region, (-2:3, 10:15)) == (-1:3, 11:13)

        ghost = Region(Cell(), Ghost(-1); halo=2)
        @test_throws MethodError Stencil.indexranges(ghost, cell_axes)
    end

    @testset "Array indexing" begin
        e₁, _ = unitoffsets(Val(2))
        cells = Region(Cell(), Physical(), Physical(); halo=1)
        faces = Region(Face(1), Physical(), Physical(); halo=1)

        A = reshape(collect(1:30), 5, 6)
        @test (@inferred Base.to_indices(A, (cells,))) == (2:4, 2:5)
        @test A[cells] == A[2:4, 2:5]

        fill!(view(A, cells), 0)
        @test all(iszero, A[cells])

        pressure = reshape(collect(1.0:30.0), 5, 6)
        gradient = zeros(6, 6)
        @views @. gradient[faces] = pressure[faces + e₁ / 2] - pressure[faces - e₁ / 2]
        @test gradient[faces] == ones(4, 4)
    end
end
