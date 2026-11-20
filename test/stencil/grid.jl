struct StencilGridProperties
    x::Vec{2, Float64}
    p::Float64
end

function generated_stencil_grid(mesh::CartesianMesh{N, T}) where {N, T}
    Prop = @NamedTuple{x::Vec{N, T}, p::T}
    generate_grid(Cell(), Prop, mesh;halo=1)
end

@testset "Generated grids share placement, physical bounds and halo axes" begin
    for N in 1:3, T in (Float32, Float64), mask in 0:(1<<N)-1, halo in 0:2
        h = T(0.25)
        counts = ntuple(d -> 6-d, N)
        origin = ntuple(d -> T(d-2), N)
        bounds = ntuple(d -> (origin[d], origin[d]+counts[d]*h), N)
        mesh = CartesianMesh(T, h, bounds...)
        Prop = @NamedTuple{x::Vec{N, T}, p::T, v::Vec{N, T}, active::Bool}
        location = Tesserae.Stencil.Location(UInt(mask))
        grid = generate_grid(location, Prop, mesh;halo)
        region = Region(grid)
        sizes = ntuple(d -> counts[d]+Int(!iszero(mask & (1<<(d-1)))), N)
        @test grid isa Grid
        @test region == Region(location, mesh)
        @test size(region) == sizes
        @test size(grid) == map(n -> n+2halo, sizes)
        @test axes(grid.x) == axes(grid.p) == axes(grid.v) == axes(grid.active) == axes(grid)
        @test map(first, axes(grid)) == ntuple(_ -> 1-halo, N)
        @test map(last, axes(grid)) == map(n -> n+halo, sizes)
        @test eltype(grid) == Prop
        @test all(iszero, grid.p)
        @test all(iszero, grid.v)
        @test !any(grid.active)
        @test spacing(grid.x) === h
        @test volume(grid.x) ≈ prod(counts)*h^N
        for I in CartesianIndices(axes(grid))
            expected = Vec(ntuple(N) do d
                phase = iszero(mask & (1<<(d-1))) ? T(0.5) : zero(T)
                origin[d]+(I[d]-1+phase)*h
            end)
            @test grid.x[I] == expected
        end
        @test size(view(grid.p, region)) == sizes
        @test collect(coordinates(region, ntuple(_ -> h, N);origin)) == Tuple.(grid.x[region])
        @stencil begin
            grid.p[region] = sum(grid.x[region])
            grid.v[region] = grid.x[region]
        end
        @test grid.p[region] == sum.(grid.x[region])
        @test grid.v[region] == grid.x[region]
        clone = copy(grid)
        @test Region(clone) == region
        @test clone.x == grid.x
        @test clone.p == grid.p
        @test clone.p !== grid.p
        fillzero!(grid)
        @test grid.x == clone.x
        @test all(iszero, grid.p)
        @test all(iszero, grid.v)
        @test size(mesh) == map(n -> n+1, counts)
        @test mesh[ntuple(_ -> 1, N)...] == Vec(origin)
        explicit = generate_grid(Array, location, Prop, mesh;halo)
        @test Region(explicit) == region
        @test explicit.x == grid.x
        @test axes(explicit) == axes(grid)
    end
end

@testset "Named placements, boundaries and shape contracts" begin
    mesh = CartesianMesh(0.2, (-1, 1), (2, 3), (0, 0.6))
    Prop = @NamedTuple{x::Vec{3, Float64}, p::Float64}
    cells = generate_grid(Cell(), Prop, mesh;halo=1)
    C = Region(cells)
    for location in (Face(1), Face(2), Face(3), Edge(1), Edge(2), Edge(3), Vertex())
        grid = generate_grid(location, Prop, mesh;halo=1)
        R = Region(grid)
        @test R == Region(location, C)
        @test Region(Cell(), R) == C
        for d in 1:3
            aligned = Tesserae.Stencil.isnodealigned(R.location, d)
            aligned || continue
            L, U = boundaries(R, d)
            lower = getindex.(grid.x[L], d)
            upper = getindex.(grid.x[U], d)
            @test all(x -> isapprox(x, first(mesh.axes[d]);atol=8eps(Float64)), lower)
            @test all(x -> isapprox(x, last(mesh.axes[d]);atol=8eps(Float64)), upper)
        end
    end
    regular = generate_grid(Prop, mesh)
    @test_throws ArgumentError Region(regular)
    @test_throws ArgumentError generate_grid(Cell(), Prop, mesh;halo=-1)
    @test_throws ArgumentError generate_grid(Face(4), Prop, mesh)
    @test_throws Exception generate_grid(Cell(), @NamedTuple{x::Vec{2, Float64}, p::Float64}, mesh)
    @test_throws Exception generate_grid(Cell(), @NamedTuple{x::Vec{3, Float32}, p::Float64}, mesh)
    @test_throws Exception generate_grid(Cell(), @NamedTuple{x::Vec{3, Float64}, p::Vector{Float64}}, mesh)
end

@testset "Grid coordinates preserve nonbinary spacing and property structs" begin
    for T in (Float32, Float64)
        mesh = CartesianMesh(T, 0.1, (-0.3, 0.7), (1.1, 1.6))
        Prop = @NamedTuple{X::Vec{2, T}, p::T}
        for location in (Cell(), Face(1), Face(2), Vertex())
            grid = generate_grid(location, Prop, mesh;halo=1)
            R = Region(grid)
            expected = collect(coordinates(R, (spacing(mesh), spacing(mesh));origin=map(first, mesh.axes)))
            @test maximum(norm.(grid.X[R].-Vec.(expected))) ≤ 16eps(T)*maximum(abs, map(last, mesh.axes))
            @test Tesserae.get_mesh(grid) === grid.X
        end
    end
    mesh = CartesianMesh(0.25, (0, 1), (0, 1))
    grid = @inferred generated_stencil_grid(mesh)
    @test (@inferred Region(grid)) == Region(Cell(), mesh)
    custom = generate_grid(Cell(), StencilGridProperties, mesh;halo=1)
    @test custom isa Grid
    @test eltype(custom) == StencilGridProperties
    @test Region(custom) == Region(grid)
    @test all(iszero, custom.p)
end

@testset "Public regionindices follows array indexing" begin
    mesh = CartesianMesh(0.25, (0, 1), (0, 1))
    C = Region(Cell(), mesh)
    ex, ey = unitoffsets(C)
    for (A, origin) in ((reshape(1:16, 4, 4), (1, 1)),
                        (OffsetArray(reshape(1:16, 4, 4), -3:0, 7:10), (-3, 7)),
                        (zeros(C; halo=2), (1, 1)))
        x, y = origin
        cases = ((C, (x:x+3, y:y+3)),
                 (C+ex, (x+1:x+4, y:y+3)),
                 (reflect(C, first(boundaries(mesh, 1))), (x-1:-1:x-4, y:y+3)),
                 (intersect(C, C+10ex), (1:0, y:y+3)))
        for (R, expected) in cases
            indices = @inferred regionindices(A, R)
            @test indices == expected
            if checkbounds(Bool, A, expected...)
                @test view(A, R) == view(A, expected...)
            else
                @test_throws BoundsError view(A, R)
            end
        end
    end
    @test_throws DimensionMismatch regionindices(zeros(3, 4), C)
end

@testset "Logical halo storage at every placement" begin
    for N in 1:3, mask in 0:(1<<N)-1, width in 0:2
        cells = ntuple(d -> 5-d, N)
        C = Region(Cell(), cells)
        R = Region(Tesserae.Stencil.Location(UInt(mask)), C)
        values = reshape(collect(1:prod(size(R))), size(R))
        A = zeros(Int, R;halo=width)
        @test size(A) == map(n -> n+2width, size(R))
        @test axes(A) == ntuple(d -> (1-width):(size(R, d)+width), N)
        @test A isa (width == 0 ? Array : OffsetArray)
        @stencil A[R] = values
        @test A[R] == values
        @test Array(view(A, interior(R))) == values[interior(R)]
        @test eltype(zeros(Float32, R;halo=width)) == Float32
        @test fill(7, R;halo=width) == fill!(similar(A), 7)
        B = similar(A, Float32, R;halo=width)
        @test axes(B) == axes(A)
        @test eltype(B) == Float32
        @test parent(B) isa Array
        @test axes(similar(A, R;halo=width)) == axes(A)
        @test size(similar(A, R)) == size(R)
    end
    C = Region(Cell(), (4, 3))
    @test_throws ArgumentError zeros(C;halo=-1)
    @test_throws ArgumentError fill(1, C;halo=-1)
    @test_throws ArgumentError similar(zeros(1), C;halo=-1)
    @test_throws ArgumentError zeros(first(boundaries(Region(Face(1), C), 1));halo=1)
end

@testset "Even and odd reflections preserve physical values" begin
    for N in 1:3, mask in 0:(1<<N)-1, width in 1:2, odd_axes in 0:(1<<N)-1
        counts = ntuple(d -> 5-d, N)
        mesh = CartesianMesh(1, ntuple(d -> (0, counts[d]), N)...)
        R = Region(Tesserae.Stencil.Location(UInt(mask)), mesh)
        values = reshape(collect(1:prod(size(R))), size(R))
        parity = ntuple(d -> iszero(odd_axes & (1<<(d-1))) ? 1 : -1, N)
        A = fill(-999, R;halo=width)
        @stencil A[R] = values
        expected = similar(parent(A))
        for (k, I) in enumerate(CartesianIndices(axes(A)))
            reflected = ntuple(N) do d
                n, j = size(R, d), I[d]
                aligned = !iszero(mask & (1<<(d-1)))
                j < 1 ? (aligned ? 2-j : 1-j) : j > n ? (aligned ? 2n-j : 2n+1-j) : j
            end
            sign = prod(1 ≤ I[d] ≤ size(R, d) ? 1 : parity[d] for d in 1:N)
            expected[k] = sign*values[CartesianIndex(reflected)]
        end
        for _ in 1:2
            for d in 1:N, wall in boundaries(mesh, d)
                H = halos(A, R, wall)
                I = reflect(H, wall)
                @stencil A[H] = parity[d]*A[I]
            end
            @test parent(A) == expected
            @test A[R] == values
        end
    end
end

@testset "Halo regions follow the physical sample indices" begin
    mesh = CartesianMesh(1, (0, 4), (0, 3))
    F = Region(Face(1), mesh)
    values = reshape(collect(1:15), size(F))
    shifted = OffsetArray(values, -3:1, 7:9)
    @test shifted[F] == values
    for d in 1:2, wall in boundaries(mesh, d)
        H = halos(shifted, F, wall)
        @test size(view(shifted, H)) == ntuple(j -> j == d ? 0 : size(F, j), 2)
        @test size(view(shifted, reflect(H, wall))) == size(H)
    end

    padded = OffsetArray(reshape(collect(1:21), 7, 3), 0:6, 7:9)
    left, right = boundaries(mesh, 1)
    @test view(padded, halos(padded, F, left)) == view(padded, 0:0, 7:9)
    @test view(padded, halos(padded, F, right)) == view(padded, 6:6, 7:9)
    for wall in boundaries(mesh, 2)
        @test size(view(padded, halos(padded, F, wall))) == (7, 0)
    end
end

@testset "Parallel Region planes use ordinary native views" begin
    for N in 1:3, mask in 1:(1<<N)-1, halo in 0:1
        C = Region(Cell(), ntuple(_ -> 4, N))
        R = Region(Tesserae.Stencil.Location(UInt(mask)), C)
        for d in 1:N
            Tesserae.Stencil.isnodealigned(R.location, d) || continue
            A = fill(7, R;halo)
            L, U = boundaries(R, d)
            expected = copy(A)
            @views expected[L] .= 0
            @views expected[U] .= 0
            @stencil A[[L, U]] = 0
            @test A == expected
            @test size(view(A, [L, U])) == ntuple(q -> q==d ? 2 : size(R, q), N)
            @views A[L] .= 3
            @views A[U] .= 5
            @views @. A[[L, U]] = A[[U, L]]
            @test all(==(5), A[L])
            @test all(==(3), A[U])
            @test view(A, [L]) == view(A, L)
            @test view(A, [L, L]) == view(A, L)
            @test (@inferred Base.to_indices(A, ([L, U],))) == Base.to_indices(A, ([L, U],))
        end
    end
    R = Region(Vertex(), (5, 4))
    A = zeros(size(R))
    left, right = boundaries(R, 1)
    bottom, top = boundaries(R, 2)
    @test_throws ArgumentError view(A, [left, top])
    @test_throws ArgumentError view(A, [left, right, bottom])
    @test_throws ArgumentError view(A, [left, Region(Face(1), (5, 3))])
end
