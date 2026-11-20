using Tesserae, Tesserae.Stencil, Test, LinearAlgebra
using OffsetArrays

@testset "Each placement takes its own sample counts" begin
    C = Region(Cell(), (32, 24, 16))
    placements = (Cell(), Face(1), Face(2), Face(3), Edge(1), Edge(2), Edge(3), Vertex())
    expected = ((32, 24, 16), (33, 24, 16), (32, 25, 16), (32, 24, 17),
                (32, 25, 17), (33, 24, 17), (33, 25, 16), (33, 25, 17))
    for (location, samples) in zip(placements, expected)
        independent = Region(location, samples)
        derived = Region(location, C)
        @test size(independent) == samples
        @test independent == derived
        @test Region(Cell(), independent) == C
    end
    F = Region(Face(1), (33, 24))
    @test size(Region(Cell(), F)) == (32, 24)
    @test Region(Face(1), Region(Cell(), F)) == F
    @test_throws ArgumentError Region(Face(1), (1, 24))
    @test_throws ArgumentError Region(Face(3), (4, 3))
    @test_throws ArgumentError Region(Edge(3), (4, 3))
end

@testset "Composed selections on 3D placements" begin
    for placement in (Vertex(), Edge(3))
        samples = placement == Vertex() ? (5, 4, 3) : (5, 4, 2)
        R = Region(placement, samples)
        values = reshape(collect(1:prod(samples)), samples)
        left = first(boundaries(R, 1))
        selected = interior(left, 2)
        @test selected == first(boundaries(interior(R, 2), 1))
        @test values[selected] == values[1:1, 2:3, :]
        corner = last(boundaries(left, 2))
        @test values[corner] == values[1:1, 4:4, :]
        @test corner == first(boundaries(last(boundaries(R, 2)), 1))
        @test size(interior(left, 1)) == (0, 4, samples[3])
    end
    edge = first(boundaries(Region(Edge(3), (5, 4, 2)), 1))
    @test interior(edge, 3) == edge
    @test_throws ArgumentError boundaries(edge, 3)
end

@testset "Boundary and partial-region contracts" begin
    F = Region(Face(1), (5, 3))
    C = Region(Cell(), F)
    values = reshape(collect(1:12), 4, 3)
    I = interior(F)
    L, R = boundaries(F)
    ex, ey = unitoffsets(F)
    @test values[I-ex/2] == values[1:3, :]
    @test values[I+ex/2] == values[2:4, :]
    @test values[L+ex/2] == values[1:1, :]
    @test values[R-ex/2] == values[4:4, :]
    @test size(interior(C)) == (4, 3)
    @test_throws ArgumentError boundaries(C, 1)
    @test_throws ArgumentError boundaries(F, 2)
    @test_throws ArgumentError Region(Cell(), I)
    @test_throws ArgumentError Region(Cell(), L)
    @test_throws DimensionMismatch zeros(8, 7)[C]

    V = Region(Vertex(), (5, 4, 3))
    @test size(interior(V)) == (3, 2, 1)
    @test size(first(boundaries(V, 2))) == (5, 1, 3)
    @test_throws ArgumentError boundaries(V)
    E = Region(Edge(3), (5, 4, 2))
    @test size(interior(E)) == (3, 2, 2)
    @test size(last(boundaries(E, 1))) == (1, 4, 2)
    @test_throws ArgumentError boundaries(E, 3)

    singleton = Region(Cell(), (1, 3))
    empty_faces = interior(Region(Face(1), singleton))
    f = zeros(2, 3)
    @test size(view(f, empty_faces)) == (0, 3)
end

@testset "Cross-component MAC interpolation" begin
    F = Region(Face(1), (6, 4)) # start independently from the x-face placement
    C = Region(Cell(), F)
    @test size(C) == (5, 4)
    ex, ey = unitoffsets(F)
    h = (0.1, 0.2)
    v = [2*(i-0.5)*h[1]+3*(j-1)*h[2]+1 for i in 1:5, j in 1:5]
    u = similar(v, size(F))
    I = interior(F, 1)
    @views @. u[I] = (v[I+(ex+ey)/2]+v[I+(ex-ey)/2]+
                     v[I+(-ex+ey)/2]+v[I+(-ex-ey)/2])/4
    expected = [2*i*h[1]+3*(j-0.5)*h[2]+1 for i in 1:4, j in 1:4]
    @test maximum(abs, u[I]-expected) < 2e-14
end

@testset "Offset dimensions and interior axis validation" begin
    C = Region(Cell(), (5, 4))
    ex, ey = unitoffsets(C)
    e3 = first(unitoffsets(Region(Cell(), (5, 4, 3))))
    @test_throws MethodError C+e3
    @test_throws MethodError C-e3
    @test_throws ArgumentError interior(C, 0)
    @test_throws ArgumentError interior(C, 3)
    @test_throws MethodError interior(C, ex)
    @test_throws MethodError interior(C, (-ex, ex))
end

@testset "Placement follows half-cell shifts without a dimension-specific table" begin
    for N in 1:3, mask in 0:(1<<N)-1
        C = Region(Cell(), ntuple(_ -> 4, N))
        location = Tesserae.Stencil.Location(UInt(mask))
        offsets = unitoffsets(C)
        for d in 1:N
            shifted = location+offsets[d]/2
            target_mask = mask ⊻ (1<<(d-1))
            target = Tesserae.Stencil.Location(UInt(target_mask))
            @test Region(shifted, C) == Region(target, C)
            @test Region(location-offsets[d]/2, C) == Region(target, C)
            @test Region(location+offsets[d], C) == Region(location, C)
            @test Region(offsets[d]/2+location, C) == Region(target, C)
        end
    end
end

@testset "Flux balance including boundary cells" begin
    n = (7, 5)
    h = (0.2, 0.3)
    dt = 0.0001
    T = [1.0+0.1i+0.2j for i in 1:n[1], j in 1:n[2]]
    before = sum(T)*prod(h)
    rate = zero(T)
    C = Region(Cell(), n)
    offsets = unitoffsets(C)
    q = [similar(T, size(Region(Face(d), C))) for d in 1:2]
    for d in 1:2
        F = Region(Face(d), C)
        I = interior(F)
        L, R = boundaries(F, d)
        f, e, hd = q[d], offsets[d], h[d]
        @views @. f[I] = -(T[I+e/2]-T[I-e/2])/hd
        if d == 1
            @views @. f[L] = -(T[L+e/2]-2)/(hd/2)
            @views @. f[R] = -(3-T[R-e/2])/(hd/2)
        else
            fill!(view(f, L), 0.0)
            fill!(view(f, R), 0.4)
        end
        @views @. rate[C] -= (f[C+e/2]-f[C-e/2])/hd
    end
    @. T += dt*rate
    outward = sum((sum(selectdim(q[d], d, n[d]+1))-sum(selectdim(q[d], d, 1)))*
                  prod(h)/h[d] for d in 1:2)
    @test abs(sum(T)*prod(h)-before+dt*outward) < 1e-14
end

include("stencil/geometry.jl")
include("stencil/grid.jl")
include("stencil/broadcast.jl")
include("stencil/matrix.jl")
