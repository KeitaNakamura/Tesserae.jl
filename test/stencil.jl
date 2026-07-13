using Tesserae.Stencil

@testset "Stencil" begin
    @testset "GridOffset" begin
        e₁, e₂, e₃ = @inferred unitoffsets(Val(3))

        @test e₁ != e₂ && e₂ != e₃ && e₁ != e₃

        @test +e₁ === e₁
        @test -(-e₁) == e₁
        @test (@inferred(e₁ + e₂)) == e₂ + e₁
        @test (@inferred(e₁ - e₁)) == zero(e₁)
        @test (@inferred zero(e₁)) == e₁ - e₁
        @test !iszero(e₁)

        @test (@inferred 3e₁) == e₁ + e₁ + e₁
        @test (@inferred e₁ * 3) == e₁ + e₁ + e₁
        half = @inferred e₁ / 2
        @test half + half == e₁
        @test 3e₁ / 2 - e₁ == e₁ / 2

        @test sprint(show, e₁) == "GridOffset(1, 0, 0)"
        @test sprint(show, e₁ / 2) == "GridOffset(1//2, 0, 0)"

        @test_throws ArgumentError e₁ / 4
        @test_throws DivideError e₁ / 0
    end

    @testset "Locations" begin
        e₁, e₂, e₃ = unitoffsets(Val(3))
        physical_axes = (physical, physical, physical)
        ncells = (3, 4, 5)
        halowidth = 1
        A = zeros(ntuple(d -> ncells[d] + 2 * halowidth, Val(3)))

        cases = (
            (Cell(), zero(e₁), (2:4, 2:5, 2:6)),
            (Face(1), e₁ / 2, (2:5, 2:5, 2:6)),
            (Face(2), e₂ / 2, (2:4, 2:6, 2:6)),
            (Face(3), e₃ / 2, (2:4, 2:5, 2:7)),
            (Edge(1), (e₂ + e₃) / 2, (2:4, 2:6, 2:7)),
            (Edge(2), (e₁ + e₃) / 2, (2:5, 2:5, 2:7)),
            (Edge(3), (e₁ + e₂) / 2, (2:5, 2:6, 2:6)),
            (Vertex(), (e₁ + e₂ + e₃) / 2, (2:5, 2:6, 2:7)),
        )

        for (location, offset, expected_indices) in cases
            region = Region(location, physical_axes; halowidth) + offset
            @test parentindices(view(A, region)) == expected_indices
        end

        region = Region(Face(1), physical_axes; halowidth) + e₁ / 2
        @test size(@inferred(view(A, region))) == (4, 4, 5)
    end

    @testset "Region" begin
        e₁, e₂ = unitoffsets(Val(2))
        scalar = @inferred Region(Cell(), physical, physical; halowidth=2)
        tuple = @inferred Region(Cell(), (physical, physical); halowidth=(2, 2))
        full_region = @inferred Region(Cell(), full, full; halowidth=2)
        tuple_full = @inferred Region(Cell(), (lowhalo, full); halowidth=2)
        A = zeros(10, 10)

        @test instances(AxisRegion) == (full, physical, lowhalo, highhalo, lowboundary, highboundary)
        @test sprint(show, full) == "full"
        @test sprint(show, physical) == "physical"
        @test sprint(show, lowhalo) == "lowhalo"
        @test sprint(show, highboundary) == "highboundary"
        @test_throws MethodError Region(Cell(), :; halowidth=2)

        @test parentindices(view(A, scalar)) == parentindices(view(A, tuple))
        @test parentindices(view(A, full_region)) == (1:10, 1:10)
        @test parentindices(view(A, tuple_full)) == (1:2, 1:10)
        @test Stencil.regionranges(full_region, (-2:7, 10:19)) == (-2:7, 10:19)
        shifted = @inferred scalar + e₁
        left_shifted = @inferred e₁ + scalar
        @test shifted isa Stencil.ShiftedRegion
        @test parent(shifted) === scalar
        @test parentindices(view(A, shifted)) == (4:9, 3:8)
        @test parentindices(view(A, left_shifted)) == (4:9, 3:8)
        unshifted = @inferred shifted - e₁
        @test unshifted isa Stencil.ShiftedRegion
        @test parent(unshifted) === scalar
        @test parentindices(view(A, unshifted)) == parentindices(view(A, scalar))
        @test_throws ArgumentError e₁ - scalar
        @test_throws ArgumentError e₁ - shifted

        e, = unitoffsets(Val(1))
        cell_data = collect(1:10)
        face_data = collect(1:11)
        cells = Region(Cell(), physical; halowidth=2)
        faces = Region(Face(1), physical; halowidth=2)

        cases = (
            (Cell(), cell_data, (1:2, 3:8, 9:10)),
            (Face(1), face_data, (1:2, 3:9, 10:11)),
        )

        for (location, array, expected_indices) in cases
            low_indices, physical_indices, high_indices = expected_indices
            low = Region(location, lowhalo; halowidth=2)
            physical_region = Region(location, physical; halowidth=2)
            high = Region(location, highhalo; halowidth=2)
            low_boundary = Region(location, lowboundary; halowidth=2)
            high_boundary = Region(location, highboundary; halowidth=2)

            @test parentindices(view(array, low)) == (low_indices,)
            @test parentindices(view(array, physical_region)) == (physical_indices,)
            @test parentindices(view(array, high)) == (high_indices,)
            @test parentindices(view(array, low_boundary)) == (first(physical_indices):first(physical_indices),)
            @test parentindices(view(array, high_boundary)) == (last(physical_indices):last(physical_indices),)
            @test [array[low]; array[physical_region]; array[high]] == array
        end

        boundary = Region(Face(1), lowboundary; halowidth=2)
        cases = (
            (face_data, cells + e / 2, 4:9),
            (face_data, cells - e / 2, 3:8),
            (cell_data, faces + e / 2, 3:9),
            (cell_data, faces - e / 2, 2:8),
            (cell_data, cells + e, 4:9),
            (cell_data, cells - e, 2:7),
            (cell_data, boundary + e / 2, 3:3),
            (cell_data, boundary - e / 2, 2:2),
        )

        for (array, region, expected) in cases
            @test array[region] == array[expected]
        end

        mixed = Region(Face(1), lowhalo, highboundary; halowidth=2)
        mixed_view = @inferred view(zeros(11, 10), mixed)
        @test parentindices(mixed_view) == (1:2, 8:8)

        anisotropic = Region(Cell(), lowhalo, highhalo; halowidth=(1, 2))
        anisotropic_view = @inferred view(zeros(8, 10), anisotropic)
        @test parentindices(anisotropic_view) == (1:1, 9:10)

        shifted = Region(Face(1), physical, physical; halowidth=1) + (e₁ - e₂) / 2
        @test (@inferred Stencil.regionranges(shifted, (-2:3, 10:15))) == (-1:3, 11:13)

        function runtimewallregion(axis::Int)
            axes = ntuple(d -> d == axis ? lowhalo : full, Val(3))
            Region(Cell(), axes; halowidth=1)
        end

        runtime_axis = Base.inferencebarrier(2)::Int
        runtime_wall = @inferred runtimewallregion(runtime_axis)
        runtime_ranges = @inferred Stencil.regionranges(runtime_wall, axes(zeros(6, 6, 6)))
        @test runtime_wall isa Region{3}
        @test runtime_ranges isa NTuple{3, UnitRange{Int}}
    end

    @testset "Finite differences" begin
        e₁, e₂ = unitoffsets(Val(2))
        cells = Region(Cell(), physical, physical; halowidth=1)
        xfaces = Region(Face(1), physical, physical; halowidth=1)
        yfaces = Region(Face(2), physical, physical; halowidth=1)

        p = [i^2 + 3j^2 for i in 1:5, j in 1:4]
        ∂p∂x = zeros(Int, 6, 4)
        ∂p∂y = zeros(Int, 5, 5)

        @views @. ∂p∂x[xfaces] = p[xfaces + e₁ / 2] - p[xfaces - e₁ / 2]
        @views @. ∂p∂y[yfaces] = p[yfaces + e₂ / 2] - p[yfaces - e₂ / 2]

        expected_∂p∂x = zeros(Int, 6, 4)
        expected_∂p∂x[2:5, 2:3] .= [3 3; 5 5; 7 7; 9 9]
        expected_∂p∂y = zeros(Int, 5, 5)
        expected_∂p∂y[2:4, 2:4] .= [9 15 21; 9 15 21; 9 15 21]

        @test ∂p∂x == expected_∂p∂x
        @test ∂p∂y == expected_∂p∂y

        u = [2i for i in 1:6, _ in 1:4]
        v = [5j for _ in 1:5, j in 1:5]
        divergence = zeros(Int, 5, 4)

        @views @. divergence[cells] =
            u[cells + e₁ / 2] - u[cells - e₁ / 2] +
            v[cells + e₂ / 2] - v[cells - e₂ / 2]

        expected_divergence = zeros(Int, 5, 4)
        expected_divergence[2:4, 2:3] .= 7
        @test divergence == expected_divergence

        laplacian = zeros(Int, 5, 4)

        @views @. laplacian[cells] =
            ∂p∂x[cells + e₁ / 2] - ∂p∂x[cells - e₁ / 2] +
            ∂p∂y[cells + e₂ / 2] - ∂p∂y[cells - e₂ / 2]

        expected_laplacian = zeros(Int, 5, 4)
        expected_laplacian[2:4, 2:3] .= 8
        @test laplacian == expected_laplacian
    end

    @testset "Reflection" begin
        cells = Region(Cell(), physical, physical; halowidth=2)
        cell_low = Region(Cell(), lowhalo, physical; halowidth=2)
        cell_high = Region(Cell(), highhalo, physical; halowidth=2)
        full_cell_low = Region(Cell(), lowhalo, full; halowidth=2)

        A = zeros(Int, 8, 6)
        @test parentindices(@inferred(view(A, reflect(full_cell_low, 1)))) == (4:-1:3, 1:6)
        A[cells] .= [11 12; 21 22; 31 32; 41 42]

        @views @. A[cell_low] = A[reflect(cell_low, 1)]
        @views @. A[cell_high] = A[reflect(cell_high, 1)]

        expected_A = zeros(Int, 8, 6)
        expected_A[:, 3:4] .= [21 22; 11 12; 11 12; 21 22; 31 32; 41 42; 41 42; 31 32]
        @test A == expected_A

        reflected_wall = @inferred view(A, reflect(cell_low, 1))
        @test parentindices(reflected_wall) == (4:-1:3, 3:4)

        faces = Region(Face(1), physical, physical; halowidth=2)
        face_low = Region(Face(1), lowhalo, physical; halowidth=2)
        face_high = Region(Face(1), highhalo, physical; halowidth=2)

        F = zeros(Int, 9, 6)
        F[faces] .= [11 12; 21 22; 31 32; 41 42; 51 52]

        @views @. F[face_low] = F[reflect(face_low, 1)]
        @views @. F[face_high] = F[reflect(face_high, 1)]

        expected_F = zeros(Int, 9, 6)
        expected_F[:, 3:4] .= [31 32; 21 22; 11 12; 21 22; 31 32; 41 42; 51 52; 41 42; 31 32]
        @test F == expected_F

        corner = Region(Cell(), lowhalo, highhalo; halowidth=2)
        C = [10i + j for i in 1:8, j in 1:7]
        x_then_y = reflect(reflect(corner, 1), 2)
        y_then_x = reflect(reflect(corner, 2), 1)

        @test parentindices(view(C, x_then_y)) == (4:-1:3, 5:-1:4)
        @test parentindices(view(C, y_then_x)) == (4:-1:3, 5:-1:4)
        @test C[x_then_y] == C[y_then_x]
        expected_C = copy(C)
        expected_C[1:2, 6:7] .= [45 44; 35 34]
        @views @. C[corner] = C[x_then_y]
        @test C == expected_C

        C = [10i + j for i in 1:8, j in 1:7]
        reflectionview = (array, region, d) -> view(array, reflect(region, d))
        one_axis = @inferred reflectionview(C, corner, 1)

        @test parentindices(one_axis) == (4:-1:3, 6:1:7)
        @test one_axis == [46 47; 36 37]

        runtime_axis = Base.inferencebarrier(2)::Int
        runtime_reflection = @inferred reflect(corner, runtime_axis)
        mapped_ranges = @inferred Stencil.mappedranges(runtime_reflection, axes(C))
        @test mapped_ranges isa NTuple{2, StepRange{Int, Int}}

        mixed = Region(Cell(), lowhalo, physical, highboundary, full; halowidth=1)
        mixed_axes = axes(zeros(5, 5, 5, 5))
        @test (@inferred Stencil.regionranges(mixed, mixed_axes)) isa NTuple{4, UnitRange{Int}}
        @test (@inferred Stencil.mappedranges(reflect(mixed, 1), mixed_axes)) isa NTuple{4, StepRange{Int, Int}}

        unreflected = @inferred view(C, reflect(reflect(corner, 1), 1))
        @test parentindices(unreflected) == (1:1:2, 6:1:7)

        boundary = Region(Cell(), lowhalo, highboundary; halowidth=2)
        reflected_boundary = @inferred view(C, reflect(boundary, 1))
        @test parentindices(reflected_boundary) == (4:-1:3, 5:5)

        reflected_accumulation = [2, 3, 10, 20, 30, 40, 0, 0]
        low_halo = Region(Cell(), lowhalo; halowidth=2)
        @views @. reflected_accumulation[reflect(low_halo, 1)] += reflected_accumulation[low_halo]
        @test reflected_accumulation == [2, 3, 13, 22, 30, 40, 0, 0]

        for (location, n) in ((Cell(), 4), (Face(1), 5)), halo_axis in (lowhalo, highhalo)
            data = collect(1:n)
            halo = Region(location, halo_axis; halowidth=0)
            reflected = @inferred view(data, reflect(halo, 1))

            @test isempty(reflected)
            @views @. data[halo] = data[reflect(halo, 1)]
            @test data == collect(1:n)
        end

        e, = unitoffsets(Val(1))
        shifted_data = collect(1:9)
        shifted_halo = Region(Cell(), lowhalo; halowidth=2) + e / 2
        @test shifted_data[shifted_halo] == shifted_data[2:3]
        @test_throws ArgumentError reflect(shifted_halo, 1)
        @test_throws ArgumentError reflect(Region(Cell(), lowhalo; halowidth=2) + zero(e), 1)

        _, e₂ = unitoffsets(Val(2))
        transverse_shift = Region(Cell(), lowhalo, physical; halowidth=2) + e₂ / 2
        @test_throws ArgumentError reflect(transverse_shift, 1)
    end

    @testset "Invalid geometry" begin
        nbits = 8 * sizeof(UInt)

        for constructor in (Face, Edge), d in (-1, 0, nbits + 1)
            @test_throws ArgumentError constructor(d)
        end

        for location in (Face(3), Edge(3))
            @test_throws ArgumentError Region(location, physical, physical; halowidth=1)
        end

        @test_throws ArgumentError Region(Cell(), physical; halowidth=-1)
        @test_throws ArgumentError Region(Cell(), physical, physical; halowidth=(1, -1))
        @test_throws DimensionMismatch view(zeros(4), Region(Cell(), physical; halowidth=2))
        @test_throws DimensionMismatch view(zeros(5), Region(Face(1), physical; halowidth=2))
        @test_throws DimensionMismatch view(zeros(4), Region(Cell(), physical; halowidth=typemax(Int)))

        @test parentindices(view(zeros(5), Region(Cell(), physical; halowidth=2))) == (3:3,)
        @test parentindices(view(zeros(6), Region(Face(1), physical; halowidth=2))) == (3:4,)
        @test_throws DimensionMismatch view(zeros(5), reflect(Region(Cell(), lowhalo; halowidth=2), 1))
        @test_throws DimensionMismatch view(zeros(6), reflect(Region(Face(1), lowhalo; halowidth=2), 1))

        @test parentindices(view(zeros(6), reflect(Region(Cell(), lowhalo; halowidth=2), 1))) == (4:-1:3,)
        @test parentindices(view(zeros(7), reflect(Region(Face(1), lowhalo; halowidth=2), 1))) == (5:-1:4,)

        zero_halo = Region(Cell(), physical; halowidth=0)
        @test parentindices(view(zeros(4), zero_halo)) == (1:4,)
        @test isempty(view(zeros(4), Region(Cell(), lowhalo; halowidth=0)))

        @test_throws ArgumentError reflect(Region(Cell(), physical; halowidth=1), 1)
        @test_throws ArgumentError reflect(Region(Face(1), lowboundary; halowidth=1), 1)
        @test_throws ArgumentError reflect(Region(Cell(), lowhalo; halowidth=1), 0)
        @test_throws ArgumentError reflect(Region(Cell(), lowhalo; halowidth=1), 2)
        @test_throws ArgumentError reflect(reflect(Region(Cell(), lowhalo; halowidth=1), 1), 2)
    end
end
