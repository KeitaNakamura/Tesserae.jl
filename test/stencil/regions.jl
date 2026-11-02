@testset "Each placement takes its own sample counts" begin
    C = Region(Cell(), (32, 24, 16))
    placements = (Cell(), Face(1), Face(2), Face(3), Edge(1), Edge(2), Edge(3), Vertex())
    expected = ((32,24,16), (33,24,16), (32,25,16), (32,24,17),
                (32,25,17), (33,24,17), (33,25,16), (33,25,17))
    for (location, samples) in zip(placements, expected)
        independent = Region(location, samples)
        derived = Region(location, C)
        @test size(independent) == samples
        @test independent == derived
        @test Region(Cell(), independent) == C
        @test size(similar(zeros(1), size(independent))) == samples
    end
    F = Region(Face(1), (33, 24))
    @test size(Region(Cell(), F)) == (32,24)
    @test Region(Face(1), Region(Cell(), F)) == F
    @test_throws ArgumentError Region(Face(1), (1,24))
    @test_throws ArgumentError Region(Face(3), (4,3))
    @test_throws ArgumentError Region(Edge(3), (4,3))
end

@testset "Standard allocation and direction-specific interiors" begin
    V = Region(Vertex(), (5,4,3))
    values = reshape(collect(1:60),5,4,3)
    @test values[interior(V,1)] == values[2:4,:,:]
    @test values[interior(V,2)] == values[:,2:3,:]
    @test values[interior(V,3)] == values[:,:,2:2]
    @test values[interior(interior(V,1),2)] == values[2:4,2:3,:]
    @test interior(interior(V,1),1) == interior(V,1)
    @test foldl(interior, 1:3; init=V) == interior(V)

    E = Region(Edge(3), (5,4,2))
    @test size(interior(E,1)) == (3,4,2)
    @test size(interior(E,2)) == (5,2,2)
    @test interior(E,3) == E
    C = Region(Cell(), (4,3,2))
    @test all(d -> interior(C,d) == C, 1:3)

    F = Region(Face(1), (5,3))
    @test interior(F,1) == interior(F)
    @test interior(F,2) == F
    L,R = boundaries(F,1)
    @test interior(L,2) == L
    @test_throws ArgumentError interior(F,0)
    @test_throws ArgumentError interior(F,3)

    prototype = zeros(Float32,5,3)
    buffer = similar(prototype, size(interior(F,1)))
    @test size(buffer) == (3,3)
    @test eltype(buffer) == Float32
    @test buffer isa Matrix{Float32}
    @test_throws MethodError similar(prototype,F)
end

@testset "Composed physical-boundary selections" begin
    V = Region(Vertex(), (5,4))
    values = reshape(collect(1:20),5,4)
    L,R = boundaries(V,1)
    J = interior(L,2)
    K = interior(R,2)
    J2,K2 = boundaries(interior(V,2),1)
    @test J == J2
    @test K == K2
    @test values[J] == values[1:1,2:3]
    @test values[K] == values[5:5,2:3]
    @test size(J) == (1,2)
    @test parent(view(values,J)) === values
    @test interior(J,2) == J

    bottom,top = boundaries(L,2)
    @test values[bottom] == values[1:1,1:1]
    @test values[top] == values[1:1,4:4]
    @test bottom == first(boundaries(first(boundaries(V,2)),1))
    @test top == first(boundaries(last(boundaries(V,2)),1))

    # Selecting the physical boundary again must not expand the subset.
    repeated,opposite = boundaries(J,1)
    @test repeated == J
    @test size(opposite) == (0,2)
    @test values[opposite] == values[1:0,2:3]
    lower,upper = boundaries(interior(V,1),1)
    @test size(lower) == (0,4)
    @test size(upper) == (0,4)
    @test values[lower] == values[1:0,:]
    @test values[upper] == values[1:0,:]

    empty = interior(L,1)
    @test size(empty) == (0,4)
    @test values[empty] == values[1:0,:]
    @test interior(empty,1) == empty
    @test size(interior(L)) == (0,2)
    @test all(r -> size(r) == (0,1), boundaries(empty,2))

    # The same composed selection works in ordinary views and broadcasts.
    result = fill(-1,size(values))
    expected = copy(result)
    expected[1:1,2:3] .= 2 .* values[1:1,2:3]
    @views @. result[J] = 2*values[J]
    @test result == expected
    @views @. result[empty] = values[empty]+1
    @test result == expected

    # A half-cell shift still maps the selected points to an ordinary array.
    ex,ey = unitoffsets(V)
    facevalues = reshape(collect(1:16),4,4)
    @test facevalues[J+ex/2] == facevalues[1:1,2:3]
    @test facevalues[empty+ex/2] == facevalues[1:0,:]
    @test facevalues[empty-ex/2] == facevalues[1:0,:]

    singleton = Region(Vertex(), (2,4))
    singlevalues = reshape(collect(1:8),2,4)
    singleinterior = interior(singleton,1)
    @test size(singleinterior) == (0,4)
    for side in boundaries(singleinterior,2)
        @test size(side) == (0,1)
        @test isempty(view(singlevalues,side))
    end
end

@testset "Composed selections on 3D placements" begin
    for placement in (Vertex(),Edge(3))
        samples = placement == Vertex() ? (5,4,3) : (5,4,2)
        R = Region(placement,samples)
        values = reshape(collect(1:prod(samples)),samples)
        left = first(boundaries(R,1))
        selected = interior(left,2)
        @test selected == first(boundaries(interior(R,2),1))
        @test values[selected] == values[1:1,2:3,:]
        corner = last(boundaries(left,2))
        @test values[corner] == values[1:1,4:4,:]
        @test corner == first(boundaries(last(boundaries(R,2)),1))
        @test size(interior(left,1)) == (0,4,samples[3])
    end
    edge = first(boundaries(Region(Edge(3),(5,4,2)),1))
    @test interior(edge,3) == edge
    @test_throws ArgumentError boundaries(edge,3)
end

@testset "Standard dimension queries" begin
    V = Region(Vertex(),(5,4))
    values = reshape(collect(1:20),5,4)
    L,R = boundaries(V,1)
    cases = [
        (V, values),
        (interior(V), values[2:4,2:3]),
        (L, values[1:1,:]),
        (interior(L,2), values[1:1,2:3]),
        (interior(L,1), values[1:0,:]),
        (Region(Cell(),(4,3)), zeros(4,3)),
        (Region(Face(1),(5,3)), zeros(5,3)),
        (Region(Edge(3),(5,4,2)), zeros(5,4,2)),
    ]
    for (region,reference) in cases
        shifted = region+first(unitoffsets(region))/2
        @test ndims(shifted) == ndims(region)
        @test size(region) == size(reference)
        @test size(shifted) == size(reference)
        for d in 1:5
            @test size(region,d) == size(reference,d)
            @test size(shifted,d) == size(reference,d)
        end
        @test size(region,Int32(2)) == size(reference,2)
        @test size(shifted,Int32(2)) == size(reference,2)
        @test_throws BoundsError size(region,0)
        @test_throws BoundsError size(shifted,-1)
    end
end

@testset "Placement-aware mismatch diagnostics" begin
    C = Region(Cell(),(4,3))
    ex,ey = unitoffsets(C)
    cases = [
        (C+ex/2, zeros(4,3), "requires 5 samples", "Cell()", "(0.5, 0.0)"),
        (Region(Face(1),C)-ex/2, zeros(5,3), "requires 4 samples", "Face(1)", "(-0.5, 0.0)"),
    ]
    for (region,array,expected,placement,shift) in cases
        err = try
            view(array,region)
        catch caught
            caught
        end
        @test err isa DimensionMismatch
        message = sprint(showerror,err)
        @test occursin(expected,message)
        @test occursin("source placement=$placement",message)
        @test occursin("shift=$shift cells",message)
    end
end

@testset "All center/grid-line placements and pairwise round trips" begin
    for N in 1:4
        cells = ntuple(d -> d+2, N)
        placements = [Tesserae.Stencil.Location(UInt(mask)) for mask in 0:(1<<N)-1]
        for location in placements
            samples = ntuple(d -> cells[d]+Tesserae.Stencil.isnodealigned(location,d), N)
            source = Region(location, samples)
            A = reshape(collect(1:prod(samples)), samples)
            @test A[source] == A
            @test parent(view(A,source)) === A
            for target in placements
                converted = Region(target, source)
                @test size(converted) == ntuple(d -> cells[d]+Tesserae.Stencil.isnodealigned(target,d), N)
                @test Region(location,converted) == source
            end

            I = interior(source)
            for (d,e) in enumerate(unitoffsets(source))
                target = Tesserae.Stencil.Location(xor(location.mask, one(UInt)<<(d-1)))
                target_samples = ntuple(j -> cells[j]+Tesserae.Stencil.isnodealigned(target,j), N)
                affine = [sum(j*(J[j]-1+0.5*!Tesserae.Stencil.isnodealigned(target,j))/cells[j]
                              for j in 1:N) for J in CartesianIndices(target_samples)]
                @views differences = affine[I+e/2]-affine[I-e/2]
                @test size(differences) == size(I)
                @test maximum(abs, differences .- d/cells[d]) < 2e-14
            end
        end
    end
end

@testset "Boundary and partial-region contracts" begin
    F = Region(Face(1),(5,3))
    C = Region(Cell(), F)
    values = reshape(collect(1:12),4,3)
    I = interior(F)
    L,R = boundaries(F)
    ex,ey = unitoffsets(F)
    @test values[I-ex/2] == values[1:3,:]
    @test values[I+ex/2] == values[2:4,:]
    @test values[L+ex/2] == values[1:1,:]
    @test values[R-ex/2] == values[4:4,:]
    @test size(interior(C)) == (4,3)
    @test_throws ArgumentError boundaries(C,1)
    @test_throws ArgumentError boundaries(F,2)
    @test_throws ArgumentError Region(Cell(),I)
    @test_throws ArgumentError Region(Cell(),L)
    @test_throws DimensionMismatch zeros(8,7)[C]

    V = Region(Vertex(), (5,4,3))
    @test size(interior(V)) == (3,2,1)
    @test size(first(boundaries(V,2))) == (5,1,3)
    @test_throws ArgumentError boundaries(V)
    E = Region(Edge(3), (5,4,2))
    @test size(interior(E)) == (3,2,2)
    @test size(last(boundaries(E,1))) == (1,4,2)
    @test_throws ArgumentError boundaries(E,3)

    singleton = Region(Cell(), (1,3))
    empty_faces = interior(Region(Face(1), singleton))
    f = zeros(2,3)
    @test size(view(f,empty_faces)) == (0,3)
end

@testset "Cross-component MAC interpolation" begin
    F = Region(Face(1), (6,4)) # start independently from the x-face placement
    C = Region(Cell(), F)
    @test size(C) == (5,4)
    ex,ey = unitoffsets(F)
    h = (0.1,0.2)
    v = [2*(i-0.5)*h[1]+3*(j-1)*h[2]+1 for i in 1:5,j in 1:5]
    u = similar(v,size(F))
    I = interior(F,1)
    @views @. u[I] = (v[I+(ex+ey)/2]+v[I+(ex-ey)/2]+
                     v[I+(-ex+ey)/2]+v[I+(-ex-ey)/2])/4
    expected = [2*i*h[1]+3*(j-0.5)*h[2]+1 for i in 1:4,j in 1:4]
    @test maximum(abs,u[I]-expected) < 2e-14
end
