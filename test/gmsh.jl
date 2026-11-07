using Gmsh

const GMSH_TEST_ARGS = ["-v", "0"]

mshpath(name) = joinpath(@__DIR__, "fixtures", name)
readtestmsh(name; kwargs...) = readmsh(mshpath(name); gmsh_argv=GMSH_TEST_ARGS, kwargs...)
cellcoordinates(mesh, cell) = Tuple(Tuple(mesh[i]) for i in supportnodes(mesh, cell))
nodecoordinates(mesh) = sort([Tuple(mesh[i]) for i in eachindex(mesh)])

@testset "Gmsh" begin
    @testset "reference node order for every supported element" begin
        extension = Base.get_extension(Tesserae, :TesseraeGmshExt)
        Gmsh.gmsh.initialize(GMSH_TEST_ARGS)
        try
            for (family, order, serendip) in (("Line",1,false),("Line",2,false),("Line",3,false),
                    ("Triangle",1,false),("Triangle",2,false),("Tetrahedron",1,false),("Tetrahedron",2,false),
                    ("Quadrangle",1,false),("Quadrangle",2,true),("Quadrangle",2,false),
                    ("Hexahedron",1,false),("Hexahedron",2,true),("Hexahedron",2,false))
                element_type = Gmsh.gmsh.model.mesh.getElementType(family, order, serendip)
                name, dim, _, n, coords, _ = Gmsh.gmsh.model.mesh.getElementProperties(element_type)
                shape = extension.from_gmsh_shape(name)
                permutation = extension.from_gmsh_connectivity(shape)
                nodes = Tesserae.localnodes(shape)
                expected = [nodes[j][d] for d in 1:dim, j in 1:n]
                @test reshape(coords, Int(dim), Int(n))[:, permutation] ≈ expected
            end
        finally
            Gmsh.gmsh.finalize()
        end
    end
    @testset "Coordinate dimension" begin
        @test_throws ArgumentError readtestmsh("embedded_surface.msh")
        @test !Bool(Gmsh.gmsh.isInitialized())
        for spacedim in (0, 1, 2, 4, 2.5)
            @test_throws ArgumentError readtestmsh("embedded_surface.msh"; spacedim)
        end
        surface = readtestmsh("embedded_surface.msh"; spacedim=3)["surface"]
        @test collect(surface) == [Vec(0.0,0.0,0.0), Vec(0.0,1.0,0.0), Vec(0.0,0.0,1.0)]
        points = generate_particles(@NamedTuple{x::Vec{3,Float64}, dS::Float64, n::Vec{3,Float64}},
                                    surface, generate_quadrature_rule(basis(surface)))
        weights = generate_basis_weights(surface, size(points))
        update!(weights, points, surface; measure=points.dS, normal=points.n)
        @test sum(points.dS) ≈ 0.5
        @test all(n -> n ≈ Vec(1.0,0.0,0.0), points.n)
        @test eltype(readtestmsh("square.msh")["domain"]) === Vec{2,Float64}
        @test eltype(readtestmsh("square.msh"; spacedim=3)["domain"]) === Vec{3,Float64}
    end

    meshes = readtestmsh("square.msh")

    @test sort(collect(keys(meshes))) == ["boundary", "domain"]

    domain = meshes["domain"]
    @test Tesserae.cellshape(domain) == Tesserae.Tri3()
    @test Tesserae.ncells(domain) == 2
    @test supportnodes(domain) == [1, 2, 3, 4]
    @test sort([cellcoordinates(domain, cell) for cell in cells(domain)]) == [
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    ]

    boundary = meshes["boundary"]
    @test Tesserae.cellshape(boundary) == Tesserae.Line2()
    @test Tesserae.ncells(boundary) == 2
    @test supportnodes(boundary) == [1, 2, 3, 4]
    @test sort([cellcoordinates(boundary, cell) for cell in cells(boundary)]) == [
        ((0.0, 0.0), (0.0, 1.0)),
        ((1.0, 0.0), (1.0, 1.0)),
    ]

    fields = generate_field_meshes(meshes, Order(1))
    @test fields["domain"].nodes === fields["boundary"].nodes
    @test supportnodes(fields["domain"]) == supportnodes(fields["boundary"]) == [1, 2, 3, 4]

    reoriented_boundary = readtestmsh("square.msh"; reorient_boundary=true)["boundary"]
    @test sort([cellcoordinates(reoriented_boundary, cell) for cell in cells(reoriented_boundary)]) == [
        ((0.0, 1.0), (0.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
    ]

    @test nodecoordinates(domain) == [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]

    meshes = readtestmsh("high_order.msh"; reorient_boundary=false)

    tri6 = meshes["tri6"]
    @test Tesserae.cellshape(tri6) == Tesserae.Tri6()
    @test Tesserae.ncells(tri6) == 1
    @test cellcoordinates(tri6, 1) == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0),
    )

    tet10 = meshes["tet10"]
    @test Tesserae.cellshape(tet10) == Tesserae.Tet10()
    @test Tesserae.ncells(tet10) == 1
    @test cellcoordinates(tet10, 1) == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
    )

    meshes = readtestmsh("unnamed_physical_group.msh")
    @test collect(keys(meshes)) == ["physical_group[1,7]"]
    @test Tesserae.cellshape(meshes["physical_group[1,7]"]) == Tesserae.Line2()

    @test_throws ErrorException readtestmsh("mixed_shapes.msh")
    @test !Bool(Gmsh.gmsh.isInitialized())

    @test_throws ErrorException readtestmsh("duplicate_names.msh")
    @test !Bool(Gmsh.gmsh.isInitialized())

    @test_logs (:warn, r"no matching volume face") readtestmsh("unmatched_boundary.msh"; reorient_boundary=true)
    @test !Bool(Gmsh.gmsh.isInitialized())

    @test_logs (:warn, r"multiple matching volume faces") readtestmsh("ambiguous_boundary.msh"; reorient_boundary=true)
    @test !Bool(Gmsh.gmsh.isInitialized())

    Gmsh.initialize(GMSH_TEST_ARGS; finalize_atexit=false)
    try
        readmsh(mshpath("square.msh"))
        @test Bool(Gmsh.gmsh.isInitialized())
    finally
        Gmsh.finalize()
    end
    @test !Bool(Gmsh.gmsh.isInitialized())

    mktempdir() do dir
        curve = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(1.0, 0.0))
        curve_step = joinpath(dir, "curve.step")
        @test Tesserae.NURBS.writestep(curve_step, curve) == curve_step
        @test isfile(curve_step)
        @test !Bool(Gmsh.gmsh.isInitialized())

        bottom = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(1.0, 0.0))
        top = Tesserae.NURBS.line(Vec(0.0, 1.0), Vec(1.0, 1.0))
        left = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(0.0, 1.0))
        right = Tesserae.NURBS.line(Vec(1.0, 0.0), Vec(1.0, 1.0))
        surface = Tesserae.NURBS.coons_patch(bottom, top, left, right)
        surface_step = joinpath(dir, "surface.step")
        @test Tesserae.NURBS.writestep(surface_step, surface) == surface_step
        @test isfile(surface_step)
        @test !Bool(Gmsh.gmsh.isInitialized())
    end
end
