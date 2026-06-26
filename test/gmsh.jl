using Gmsh

cellcoordinates(mesh, cell) = Tuple(Tuple(mesh[i]) for i in supportnodes(mesh, cell))

@testset "Gmsh" begin
    meshes = readmsh(joinpath(@__DIR__, "fixtures", "square.msh"))

    @test sort(collect(keys(meshes))) == ["domain", "left"]

    domain = meshes["domain"]
    @test Tesserae.cellshape(domain) == Tesserae.Tri3()
    @test Tesserae.ncells(domain) == 2
    @test sort([cellcoordinates(domain, cell) for cell in cells(domain)]) == [
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    ]

    left = meshes["left"]
    @test Tesserae.cellshape(left) == Tesserae.Line2()
    @test Tesserae.ncells(left) == 1
    @test cellcoordinates(left, 1) == ((0.0, 1.0), (0.0, 0.0))

    @test sort([Tuple(domain[i]) for i in eachindex(domain)]) == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
end
