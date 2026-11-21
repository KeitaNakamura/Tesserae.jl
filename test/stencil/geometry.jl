@testset "Planes, intersections and reflections" begin
    mesh = CartesianMesh(0.25, (-1, 1), (2, 3))
    C = Region(Cell(), mesh)
    ex, ey = unitoffsets(C)
    left, right = boundaries(mesh, 1)
    middle = AxisPlane(mesh, 1, 0)
    @test @inferred(right-left) == 8ex
    @test left+ey == left
    @test_throws ArgumentError left-first(boundaries(mesh, 2))
    @test_throws ArgumentError AxisPlane(mesh, 0, 0)
    @test_throws ArgumentError AxisPlane(mesh, 1, Inf)
    for placement in (Cell(), Face(1), Face(2), Vertex())
        R = Region(placement, mesh)
        mirror = @inferred reflect(R, middle)
        points = collect(coordinates(R))
        @test collect(coordinates(mirror)) == map(x -> (-x[1], x[2]), points)
        @test reflect(mirror, middle) == R
        for selected in (R, mirror, interior(R))
            @test intersect(selected, R) == selected
            for x in (-2, -1, -0.7, 0, 0.125, 1, 2)
                cut = AxisPlane(mesh, 1, x)
                parts = @inferred split(selected, cut)
                @test parts[2] == intersect(selected, cut)
                samples = vec(collect(coordinates(selected)))
                for (part, on_side) in zip(parts, (<(x), ==(x), >(x)))
                    @test vec(collect(coordinates(part))) == filter(p -> on_side(p[1]), samples)
                end
            end
        end
    end
    F = Region(Face(1), mesh)
    @test boundaries(F, 1) == (intersect(F, left), intersect(F, right))
    @test size(intersect(F, C), 1) == 0
    quarter = AxisPlane(mesh, 1, -1+0.25/4)
    @test reflect(C, quarter).location == Face(1)
    @test reflect(reflect(C, quarter), quarter) == C
    @test_throws ArgumentError reflect(C, AxisPlane(mesh, 1, -0.97))
    @test_throws ArgumentError quarter-left
end

@testset "Mesh and staggered coordinates" begin
    for T in (Float32, Float64)
        mesh = CartesianMesh(T, T(0.1), (T(-0.45), T(0.55)))
        left = first(boundaries(mesh, 1))
        Prop = @NamedTuple{x::Vec{1, T}, p::T}
        for placement in (Cell(), Face(1)), halo in (0, 2)
            grid = generate_grid(placement, Prop, mesh; halo)
            R = Region(grid)
            e = only(unitoffsets(R))
            @test collect(@inferred coordinates(R)) == Tuple.(grid.x[R])
            for (i, point) in enumerate(grid.x[R])
                x = only(point)
                plane = left+(i-1)*e+(placement == Cell() ? e/2 : 0e)
                @test AxisPlane(mesh, 1, x) == plane
                @test AxisPlane(mesh, 1, Float32(x)) == plane
                @test size(intersect(R, plane)) == (1,)
                @test AxisPlane(mesh, 1, prevfloat(x)) != plane
                @test AxisPlane(mesh, 1, nextfloat(x)) != plane
            end
        end
    end

    original = CartesianMesh(0.03598795907913268, (0.11124340370392428, 0.55); warn=false)
    for mesh in (CartesianMesh(LinRange(-0.3, 0.7, 11)), original[4:10])
        R = Region(Vertex(), mesh)
        @test first.(collect(coordinates(R))) == mesh.axes[1]
        left = first(boundaries(mesh, 1))
        e = only(unitoffsets(R))
        for (i, x) in enumerate(mesh.axes[1])
            @test AxisPlane(mesh, 1, x) == left+(i-1)*e
            for cut in (prevfloat(x), nextfloat(x))
                parts = split(R, AxisPlane(mesh, 1, cut))
                @test map(r -> size(r, 1), parts) == map(f -> count(f, mesh.axes[1]), (<(cut), ==(cut), >(cut)))
            end
        end
    end

    C = Region(Cell(), (3, 2))
    F = Region(Face(1), C)
    @test first(coordinates(C, (0.25f0, 0.5f0))) === (0.125f0, 0.25f0)
    @test first(coordinates(C, (1//3, 1//5))) === (1//6, 1//10)
    @test collect(coordinates(F, (2, 3); origin=(-1, 4))) ==
        [(x, y) for x in (-1, 1, 3, 5), y in (5.5, 8.5)]
    @test_throws DimensionMismatch coordinates(C, (1.0,))
    @test_throws ArgumentError coordinates(C, (0, 1))
    @test_throws ArgumentError coordinates(C, (1, 1); origin=(NaN, 0))
end

@testset "Plane precision and frame identity" begin
    mesh = CartesianMesh(1.0, (0, 4))
    R = Region(Vertex(), mesh)
    left = first(boundaries(mesh, 1))
    e = only(unitoffsets(R))
    for x in (prevfloat(1.0), nextfloat(1.0), nextfloat(0.0), -1e30, 1e30)
        p = AxisPlane(mesh, 1, x)
        @test @inferred(p+e) != p
        @test p+e-e == p
        @test (p+e/2)-p == e/2
        @test size(intersect(R, p)) == (0,)
        for (part, keep) in zip(split(R, p), (<(x), ==(x), >(x)))
            @test first.(collect(coordinates(part))) == filter(keep, mesh.axes[1])
        end
    end
    for (x, counts) in ((prevfloat(0.0), (1, 0, 4)), (nextfloat(0.0), (2, 0, 3)))
        shifted = AxisPlane(mesh, 1, x)+e
        @test map(r -> size(r, 1), split(R, shifted)) == counts
    end
    decimal = CartesianMesh(0.1, (0, 0.4))
    decimal_cut = AxisPlane(decimal, 1, 1//10)
    @test map(r -> size(r, 1), split(Region(Vertex(), decimal), decimal_cut)) == (1, 0, 4)
    @test decimal_cut != AxisPlane(decimal, 1, 0.1)
    # Both rationals round to 1.0, but define distinct off-grid planes.
    cuts = [AxisPlane(mesh, 1, (n+1)//n) for n in (2^54, 2^55)]
    @test cuts[1] != cuts[2]
    @test all(p -> map(r -> size(r, 1), split(R, p)) == (2, 0, 3), cuts)
    shifted = AxisPlane(mesh, 1, 0.0625)+e/2
    @test isequal(shifted, AxisPlane(mesh, 1, 0.5625))
    @test hash(shifted) == hash(AxisPlane(mesh, 1, 0.5625))
    @test_throws ArgumentError reflect(R, AxisPlane(mesh, 1, nextfloat(0.0)))
    @test_throws ArgumentError AxisPlane(mesh, 1, 0.6)-AxisPlane(mesh, 1, 0.1)
    far = AxisPlane(mesh, 1, typemax(Int))
    @test far+e-e == far
    @test_throws InexactError (far+e)-left

    for T in (Float32, Float64), h in (nextfloat(zero(T)), floatmax(T)/16)
        scaled = CartesianMesh(T, h, (zero(T), 4h))
        F = Region(Face(1), scaled)
        for x in scaled.axes[1]
            @test size(intersect(F, AxisPlane(scaled, 1, x))) == (1,)
        end
    end
    a = Region(Cell(), CartesianMesh(Float32, 0.1f0, (0f0, 1f0)))
    b = Region(Cell(), CartesianMesh(Float64, Float64(0.1f0), (0.0, 10Float64(0.1f0))))
    coarse = Region(Cell(), CartesianMesh(Float32, 1f0, (1f7, 1f7+4)))
    fine = Region(Cell(), CartesianMesh(1.0, (1e7, 1e7+4)))
    @test coarse.frame.axes[1] == fine.frame.axes[1]
    @test first(coordinates(coarse)) != first(coordinates(fine))
    @test coarse != fine
    dense = CartesianMesh(0.3, (1e7, 1e7+3))
    @test_throws ArgumentError reflect(Region(Cell(), dense), AxisPlane(dense, 1, 1f7+1))
    @test a != b
    @test_throws ArgumentError intersect(a, b)
    empty = intersect(a, a+20e)
    @test size(empty) == (0,)
    @test length(Set([empty, empty+e])) == 1
    @test_throws ArgumentError (@stencil zeros(10)[empty] = ones(10)[b])
end
