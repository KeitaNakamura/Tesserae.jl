using Gmsh

const GMSH_TEST_ARGS = ["-v", "0"]

mshpath(name) = joinpath(@__DIR__, "fixtures", name)
readtestmsh(name) = readmsh(mshpath(name); gmsh_argv=GMSH_TEST_ARGS)
cellcoordinates(mesh, cell) = Tuple(Tuple(mesh[i]) for i in supportnodes(mesh, cell))
nodecoordinates(mesh) = sort([Tuple(mesh[i]) for i in eachindex(mesh)])

@testset "Gmsh" begin
    meshes = readtestmsh("square.msh")

    @test sort(collect(keys(meshes))) == ["boundary", "domain"]

    domain = meshes["domain"]
    @test Tesserae.cellshape(domain) == Tesserae.Tri3()
    @test Tesserae.ncells(domain) == 2
    @test sort([cellcoordinates(domain, cell) for cell in cells(domain)]) == [
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    ]

    boundary = meshes["boundary"]
    @test Tesserae.cellshape(boundary) == Tesserae.Line2()
    @test Tesserae.ncells(boundary) == 2
    @test sort([cellcoordinates(boundary, cell) for cell in cells(boundary)]) == [
        ((0.0, 1.0), (0.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
    ]

    @test nodecoordinates(domain) == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    meshes = readtestmsh("high_order.msh")

    tri6 = meshes["tri6"]
    @test Tesserae.cellshape(tri6) == Tesserae.Tri6()
    @test Tesserae.ncells(tri6) == 1
    @test cellcoordinates(tri6, 1) == (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
    )

    tet10 = meshes["tet10"]
    @test Tesserae.cellshape(tet10) == Tesserae.Tet10()
    @test Tesserae.ncells(tet10) == 1
    @test cellcoordinates(tet10, 1) == (
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (7.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
        (9.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    )

    meshes = readtestmsh("unnamed_physical_group.msh")
    @test collect(keys(meshes)) == ["physical_group[1,7]"]
    @test Tesserae.cellshape(meshes["physical_group[1,7]"]) == Tesserae.Line2()

    @test_throws ErrorException readtestmsh("mixed_shapes.msh")
    @test !Bool(Gmsh.gmsh.isInitialized())

    @test_throws ErrorException readtestmsh("duplicate_names.msh")
    @test !Bool(Gmsh.gmsh.isInitialized())

    Gmsh.initialize(GMSH_TEST_ARGS; finalize_atexit=false)
    try
        readmsh(mshpath("square.msh"))
        @test Bool(Gmsh.gmsh.isInitialized())
    finally
        Gmsh.finalize()
    end
    @test !Bool(Gmsh.gmsh.isInitialized())
end
