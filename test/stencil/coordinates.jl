using Test

@testset "Cell and face coordinates with a nonzero origin" begin
    C = Region(Cell(), (3,2))
    Fx, Fy = Region(Face(1), C), Region(Face(2), C)
    h, origin = (2.0,3.0), (-1.0,4.0)
    expected_cell = [(x,y) for x in (0.0,2.0,4.0), y in (5.5,8.5)]
    expected_xface = [(x,y) for x in (-1.0,1.0,3.0,5.0), y in (5.5,8.5)]
    expected_yface = [(x,y) for x in (0.0,2.0,4.0), y in (4.0,7.0,10.0)]
    for (region,expected) in ((C,expected_cell), (Fx,expected_xface), (Fy,expected_yface))
        points = coordinates(region,h; origin)
        @test !(points isa AbstractArray)
        @test size(points) == size(region) == size(expected)
        @test collect(points) == expected
        @test [2x+3y for (x,y) in points] == map(x -> 2x[1]+3x[2],expected)
    end
    @test collect(coordinates(Region(Face(1),(4,2)),h; origin)) == expected_xface
    @test collect(coordinates(Region(Face(2),(3,3)),h; origin)) == expected_yface
    @test collect(coordinates(C,collect(h); origin=collect(origin))) == expected_cell
    @test first(coordinates(C,h)) == (1.0,1.5)
    @test first(coordinates(Fx,h)) == (0.0,1.5)
    @test first(coordinates(Fy,h)) == (1.0,0.0)

    L,R = boundaries(Fx)
    B,T = boundaries(Fy)
    cases = ((L,expected_xface[1:1,:]), (R,expected_xface[4:4,:]),
             (B,expected_yface[:,1:1]), (T,expected_yface[:,3:3]),
             (interior(Fx),expected_xface[2:3,:]),
             (interior(Fy),expected_yface[:,2:2]), (interior(C),expected_cell))
    for (region,expected) in cases
        points = coordinates(region,h; origin)
        @test size(points) == size(region) == size(expected)
        @test collect(points) == expected
    end

    V = Region(Vertex(), C)
    left = first(boundaries(V,1))
    strip = interior(left,2)
    corner = last(boundaries(left,2))
    @test collect(coordinates(strip,h; origin)) == [(-1.0,7.0);;]
    @test size(coordinates(strip,h; origin)) == (1,1)
    @test collect(coordinates(corner,h; origin)) == [(-1.0,10.0);;]
    @test size(coordinates(corner,h; origin)) == (1,1)

    empty = interior(L,1)
    points = coordinates(empty,h; origin)
    @test size(points) == (0,2)
    @test size(collect(points)) == (0,2)
    @test size([x+y for (x,y) in points]) == (0,2)
    @test isempty(points)
    singleton = Region(Cell(),(1,2))
    @test size(coordinates(interior(Region(Face(1),singleton)),h)) == (0,2)
end

@testset "One- and three-dimensional coordinates" begin
    C1 = Region(Cell(),(3,))
    F1 = Region(Face(1),C1)
    @test collect(coordinates(C1,(2.0,); origin=(-1.0,))) == [(0.0,),(2.0,),(4.0,)]
    @test collect(coordinates(F1,(2.0,); origin=(-1.0,))) == [(-1.0,),(1.0,),(3.0,),(5.0,)]
    @test collect(coordinates(last(boundaries(F1)),(2.0,); origin=(-1.0,))) == [(5.0,)]

    C = Region(Cell(),(2,3,2))
    h,origin = (2.0,4.0,6.0),(-1.0,2.0,-3.0)
    cases = (
        (C, [(x,y,z) for x in (0.0,2.0), y in (4.0,8.0,12.0), z in (0.0,6.0)]),
        (Region(Face(1),C), [(x,y,z) for x in (-1.0,1.0,3.0), y in (4.0,8.0,12.0), z in (0.0,6.0)]),
        (Region(Face(2),C), [(x,y,z) for x in (0.0,2.0), y in (2.0,6.0,10.0,14.0), z in (0.0,6.0)]),
        (Region(Face(3),C), [(x,y,z) for x in (0.0,2.0), y in (4.0,8.0,12.0), z in (-3.0,3.0,9.0)]),
        (Region(Edge(3),C), [(x,y,z) for x in (-1.0,1.0,3.0), y in (2.0,6.0,10.0,14.0), z in (0.0,6.0)]),
        (Region(Vertex(),C), [(x,y,z) for x in (-1.0,1.0,3.0), y in (2.0,6.0,10.0,14.0), z in (-3.0,3.0,9.0)]),
    )
    for (region,expected) in cases
        points = coordinates(region,h; origin)
        @test size(points) == size(region) == size(expected)
        @test collect(points) == expected
    end
    boundary = last(boundaries(Region(Face(2),C)))
    @test size(coordinates(boundary,h; origin)) == (2,1,2)
    @test collect(coordinates(boundary,h; origin)) ==
        [(x,14.0,z) for x in (0.0,2.0), y in 1:1, z in (0.0,6.0)]
end

@testset "Shifted coordinates agree with staggered array indexing" begin
    C = Region(Cell(),(3,2))
    Fx = Region(Face(1),C)
    ex,ey = unitoffsets(C)
    h,origin = (2.0,3.0),(-1.0,4.0)
    cellpoints = collect(coordinates(C,h; origin))
    facepoints = collect(coordinates(Fx,h; origin))
    I = interior(Fx)
    L,R = boundaries(Fx)
    for region in (I-ex/2, I+ex/2, L+ex/2, R-ex/2)
        @test size(coordinates(region,h; origin)) == size(region)
        @test collect(coordinates(region,h; origin)) == cellpoints[region]
    end
    for region in (C-ex/2, C+ex/2)
        @test collect(coordinates(region,h; origin)) == facepoints[region]
    end
    @test collect(coordinates(C+ex+ey,h; origin)) ==
        [(x+2,y+3) for (x,y) in cellpoints]
end

@testset "Coordinate precision and input checks" begin
    C = Region(Cell(),(3,2))
    points32 = coordinates(C,(0.25f0,0.5f0))
    @test first(points32) === (0.125f0,0.25f0)
    @test first(coordinates(C,(1//4,1//2))) == (1//8,1//4)
    @test_throws DimensionMismatch coordinates(C,(1.0,))
    @test_throws DimensionMismatch coordinates(C,(1.0,2.0); origin=(0.0,))
    for h in ((0.0,1.0),(-1.0,1.0),(Inf,1.0),(NaN,1.0),(1.0+0im,1.0))
        @test_throws ArgumentError coordinates(C,h)
    end
    @test_throws ArgumentError coordinates(C,(1.0,1.0); origin=(NaN,0.0))
    @test_throws ArgumentError coordinates(C,(1.0,1.0); origin=(0.0,Inf))
end
