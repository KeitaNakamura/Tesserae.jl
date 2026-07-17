@testset "CartesianMesh" begin
    # constructor
    ## default
    (@inferred CartesianMesh(Float32, 1, (0,3), (1,4)))::CartesianMesh{2, Float32}
    (@inferred CartesianMesh(Float64, 1, (0,3), (1,4), (0,2)))::CartesianMesh{3, Float64}
    mesh = (@inferred CartesianMesh(1, (0,3), (1,4), (0,2)))::CartesianMesh{3, Float64}
    @test mesh[1] === Vec(0.0,1.0,0.0)
    @test mesh[end] === Vec(3.0,4.0,2.0)
    @test mesh == map(Vec, (Iterators.product(range(0,3,step=1), range(1,4,step=1), range(0,2,step=1))))
    covered_mesh = @test_logs (:warn, r"not divisible by spacing") CartesianMesh(0.3, (0,1))
    @test size(covered_mesh) === (5,)
    @test covered_mesh[1] === Vec(0.0)
    @test covered_mesh[end] === Vec(1.2)
    quiet_covered_mesh = @test_nowarn CartesianMesh(0.3, (0,1); warn=false)
    @test quiet_covered_mesh == covered_mesh
    for n in (2, 3, 7, 10)
        L = 1.0
        exact_cover_mesh = @test_nowarn CartesianMesh(L/n, (0, L))
        @test size(exact_cover_mesh) === (n + 1,)
        @test exact_cover_mesh[end] ≈ Vec(L)
    end
    ## from ranges
    mesh2 = (@inferred CartesianMesh(range(0.0,3,step=1), range(1.0,4,step=1), range(0.0,2,step=1)))::CartesianMesh{3, Float64}
    (@inferred CartesianMesh(range(0.0f0,3,step=1), range(1.0f0,4,step=1), range(0.0f0,2,step=1)))::CartesianMesh{3, Float32}
    @test mesh == mesh2
    @test_throws MethodError CartesianMesh(range(0,3,step=1), range(1,4,step=1), range(0,2,step=1))

    # misc
    mesh = CartesianMesh(0.2, (0,3), (0,4))
    @test size(mesh) === (16,21)
    @test IndexStyle(mesh) === IndexCartesian()
    @test (@inferred spacing(mesh)) === 0.2
    @test (@inferred Tesserae.spacing_inv(mesh)) === inv(0.2)
    @test (@inferred Tesserae.block_size_log2(mesh)) === Tesserae.BLOCK_SIZE_LOG2

    mesh_block3 = CartesianMesh(0.2, (0,3), (0,4); block_size_log2=Val(3))
    @test (@inferred Tesserae.block_size_log2(mesh_block3)) === 3
    @test Tesserae.blockwidth(mesh_block3) === 8
    @test Tesserae.nblocks(mesh_block3) === (2, 3)
    @test Tesserae.findblock(Vec(1.7, 2.1), mesh_block3) === CartesianIndex(2, 2)
    @test Tesserae.block_size_log2(mesh_block3[2:10, 3:15]) === 3
    @test_throws MethodError CartesianMesh(0.2, (0,3), (0,4); block_size_log2=3)

    # isinside
    @test (@inferred Tesserae.isinside(Vec(0.1,0.3), mesh)) === true
    @test (@inferred Tesserae.isinside(Vec(0.2,0.4), mesh)) === true
    ## exactly on the boundary
    @test (@inferred Tesserae.isinside(Vec(0.0,0.0), mesh)) === true
    @test (@inferred Tesserae.isinside(Vec(3.0,4.0), mesh)) === false
    ## outside
    @test (@inferred Tesserae.isinside(Vec(-1.0,3.0), mesh)) === false
    @test (@inferred Tesserae.isinside(Vec(1.0,-3.0), mesh)) === false

    # supportnodes
    @test (@inferred supportnodes(Vec(0.1,0.1), 1, mesh)) === CartesianIndices((1:2,1:2))
    @test (@inferred supportnodes(Vec(0.3,0.1), 2, mesh)) === CartesianIndices((1:4,1:3))
    @test (@inferred supportnodes(Vec(0.1,0.3), 2, mesh)) === CartesianIndices((1:3,1:4))
    ## exactly on the node
    @test (@inferred supportnodes(Vec(0.2,0.4), 1, mesh)) === CartesianIndices((2:3,3:4))
    @test (@inferred supportnodes(Vec(0.2,0.4), 2, mesh)) === CartesianIndices((1:4,2:5))
    @test (@inferred supportnodes(Vec(3.0,4.0), 2, mesh)) === CartesianIndices((1:0,1:0))
    ## outside
    @test (@inferred supportnodes(Vec(-0.1,3.05), 3, mesh)) === CartesianIndices((1:0,1:0))

    # findcell
    @test findcell(Vec(0.1,0.1), mesh) === CartesianIndex(1,1)
    @test findcell(Vec(2.3,1.1), mesh) === CartesianIndex(12,6)
    ## exactly on the node
    @test findcell(Vec(0.0,0.0), mesh) === CartesianIndex(1,1)
    @test findcell(Vec(3.0,4.0), mesh) === nothing
end

@testset "FEMesh" begin
    cmesh = CartesianMesh(0.5, (0,2), (0,3))
    mesh = FEMesh(cmesh)
    @test length(mesh) == 35
    @test Tesserae.ncells(mesh) == 24
    @test collect(cells(mesh)) == collect(1:Tesserae.ncells(mesh))
    @test supportnodes(mesh, 1) === Tesserae.cellsupports(mesh)[1]
    @test supportnodes(mesh) === getfield(mesh, :usednodes)
    @test supportnodes(mesh) == collect(eachindex(mesh))
    @test mesh == vec(cmesh)
    cmesh′ = CartesianMesh(0.5, (1,3), (1,4))
    mesh .= vec(cmesh′) # test setindex!
    @test mesh == vec(cmesh′)
    @test Tesserae.cellshape(FEMesh(CartesianMesh(1, (0,2)))) == Tesserae.Line2()
    @test Tesserae.cellshape(FEMesh(CartesianMesh(1, (0,2), (0,3)))) == Tesserae.Quad4()
    @test Tesserae.cellshape(FEMesh(CartesianMesh(1, (0,2), (0,3), (0,4)))) == Tesserae.Hex8()

    @testset "generate_field_meshes" begin
        geometry = FEMesh(Tesserae.Quad9(), CartesianMesh(1, (0,2), (0,1)))
        # The unused first node makes compact field numbering observable.
        geometry_nodes = [Vec(-1.0, -1.0), geometry...]
        geometry_supports = map(cell -> supportnodes(geometry, cell) .+ 1, cells(geometry))
        left = FEMesh(Tesserae.Quad9(), geometry_nodes, geometry_supports[1:1])
        right = FEMesh(Tesserae.Quad9(), geometry_nodes, geometry_supports[2:2])
        bottom_face = first(Tesserae.faces(Tesserae.Quad9()))
        left_boundary = geometry_supports[1][bottom_face]
        right_boundary = geometry_supports[2][bottom_face]
        # Reverse the second boundary cell to test preservation of its local ordering.
        boundary = FEMesh(Tesserae.Line3(), geometry_nodes, [left_boundary, Tesserae.SVector(right_boundary[2], right_boundary[1], right_boundary[3])])
        geometries = (left, right, boundary)

        @testset "construction and numbering" begin
            velocity = @inferred generate_field_meshes(geometries)
            @test Tesserae.cellshape.(velocity) == (Tesserae.Quad9(), Tesserae.Quad9(), Tesserae.Line3())
            @test length.(velocity) == (15, 15, 15)
            @test velocity[1].nodes === velocity[2].nodes === velocity[3].nodes
            @test parent(velocity[1].nodes) === geometry_nodes
            @test supportnodes(velocity[1], 1) == supportnodes(geometry, 1)
            @test supportnodes(velocity[2], 1) == supportnodes(geometry, 2)

            same_order = @inferred generate_field_meshes(geometries, Order(2))
            @test Tesserae.cellshape.(same_order) == Tesserae.cellshape.(velocity)
            @test length.(same_order) == length.(velocity)
            @test map(mesh -> supportnodes.(Ref(mesh), cells(mesh)), same_order) == map(mesh -> supportnodes.(Ref(mesh), cells(mesh)), velocity)

            pressure = @inferred generate_field_meshes(geometries, Order(1))
            @test Tesserae.cellshape.(pressure) == (Tesserae.Quad4(), Tesserae.Quad4(), Tesserae.Line2())
            @test length.(pressure) == (6, 6, 6)
            @test pressure[1].nodes === pressure[2].nodes === pressure[3].nodes
            @test supportnodes(pressure[1], 1) == Tesserae.SVector(1, 2, 5, 4)
            @test supportnodes(pressure[2], 1) == Tesserae.SVector(2, 3, 6, 5)
            @test supportnodes.(Ref(pressure[3]), cells(pressure[3])) == [Tesserae.SVector(1, 2), Tesserae.SVector(3, 2)]
            @test supportnodes(pressure[1]) == [1, 2, 4, 5]
            @test supportnodes(pressure[3]) == [1, 2, 3]

            field_dict = generate_field_meshes(Dict("left" => left, "right" => right, "boundary" => boundary), Order(1))
            @test Set(keys(field_dict)) == Set(("left", "right", "boundary"))
            @test supportnodes(field_dict["left"], 1) == supportnodes(pressure[1], 1)
            @test supportnodes(field_dict["right"], 1) == supportnodes(pressure[2], 1)
            @test supportnodes.(Ref(field_dict["boundary"]), cells(field_dict["boundary"])) == supportnodes.(Ref(pressure[3]), cells(pressure[3]))
            @test field_dict["left"].nodes === field_dict["right"].nodes === field_dict["boundary"].nodes

            original_node = geometry_nodes[2]
            geometry_nodes[2] = Vec(-2.0, -2.0)
            @test velocity[1][1] == pressure[1][1] == geometry_nodes[2]
            pressure[1][1] = Vec(-3.0, -3.0)
            @test velocity[1][1] == geometry_nodes[2] == pressure[1][1]
            geometry_nodes[2] = original_node
            @test_throws ArgumentError merge!(pressure[1], pressure[1])
        end

        @testset "validation" begin
            diagonal = FEMesh(Tesserae.Line3(), geometry_nodes, [geometry_supports[1][Tesserae.SVector(1, 3, 9)]])
            @test_throws ArgumentError generate_field_meshes((left, diagonal))
            wrong_midpoint = FEMesh(Tesserae.Line3(), geometry_nodes, [Tesserae.SVector(left_boundary[1], left_boundary[2], geometry_supports[1][9])])
            @test_throws ArgumentError generate_field_meshes((left, wrong_midpoint))
            separate_boundary = FEMesh(Tesserae.Line3(), copy(geometry_nodes), supportnodes.(Ref(boundary), cells(boundary)))
            @test_throws ArgumentError generate_field_meshes((left, separate_boundary))
            @test_throws ArgumentError generate_field_meshes(())
            line4 = FEMesh(Tesserae.Line4(), [Vec(0.0), Vec(1.0), Vec(1 / 3), Vec(2 / 3)], [Tesserae.SVector(1, 2, 3, 4)])
            @test_throws ArgumentError generate_field_meshes((line4,), Order(2))
        end

        @testset "shape families and traces" begin
            for (geometry_shape, field_shape, cmesh) in (
                    (Tesserae.Line3(), Tesserae.Line2(), CartesianMesh(1, (0,1))),
                    (Tesserae.Quad9(), Tesserae.Quad4(), CartesianMesh(1, (0,1), (0,1))),
                    (Tesserae.Hex27(), Tesserae.Hex8(), CartesianMesh(1, (0,1), (0,1), (0,1))),
                    (Tesserae.Tri6(), Tesserae.Tri3(), CartesianMesh(1, (0,1), (0,1))),
                    (Tesserae.Tet10(), Tesserae.Tet4(), CartesianMesh(1, (0,1), (0,1), (0,1))),
                )
                field = only(generate_field_meshes((FEMesh(geometry_shape, cmesh),), Order(1)))
                @test Tesserae.cellshape(field) == field_shape
            end

            volume_geometry = FEMesh(Tesserae.Hex27(), CartesianMesh(1, (0,1), (0,1), (0,1)))
            surface_support = supportnodes(volume_geometry, only(cells(volume_geometry)))[first(Tesserae.faces(Tesserae.Hex27()))]
            edge_support = surface_support[first(Tesserae.faces(Tesserae.Quad9()))]
            surface_geometry = FEMesh(Tesserae.Quad9(), volume_geometry.nodes, [surface_support])
            edge_geometry = FEMesh(Tesserae.Line3(), volume_geometry.nodes, [edge_support])
            fields3d = @inferred generate_field_meshes((volume_geometry, surface_geometry, edge_geometry), Order(1))
            @test Tesserae.cellshape.(fields3d) == (Tesserae.Hex8(), Tesserae.Quad4(), Tesserae.Line2())
            @test length.(supportnodes.(fields3d)) == (8, 4, 2)
        end
    end

    function compute_volume(mesh)
        dim = Tesserae.get_dimension(Tesserae.cellshape(mesh))
        ParticleProp = @NamedTuple begin
            x      :: Vec{dim, Float64}
            detJdV :: Float64
        end
        points = generate_particles(ParticleProp, mesh)
        weights = generate_basis_weights(mesh, size(points))
        update!(weights, points, mesh; measure=points.detJdV)
        sum(points.detJdV)
    end
    cmesh = CartesianMesh(1, (0,2))
    @test compute_volume(FEMesh(Tesserae.Line2(), cmesh)) ≈
          compute_volume(FEMesh(Tesserae.Line3(), cmesh)) ≈ 2

    cmesh = CartesianMesh(1, (0,2), (-1,3))
    @test compute_volume(FEMesh(Tesserae.Quad4(), cmesh)) ≈
          compute_volume(FEMesh(Tesserae.Quad8(), cmesh)) ≈
          compute_volume(FEMesh(Tesserae.Quad9(), cmesh)) ≈
          compute_volume(FEMesh(Tesserae.Tri3(), cmesh))  ≈
          compute_volume(FEMesh(Tesserae.Tri6(), cmesh))  ≈ 8
    cmesh = CartesianMesh(1, (0,2), (-1,3), (2,5))
    @test compute_volume(FEMesh(Tesserae.Hex8(), cmesh))  ≈
          compute_volume(FEMesh(Tesserae.Hex20(), cmesh)) ≈
          compute_volume(FEMesh(Tesserae.Hex27(), cmesh)) ≈
          compute_volume(FEMesh(Tesserae.Tet4(), cmesh))  ≈
          compute_volume(FEMesh(Tesserae.Tet10(), cmesh)) ≈ 24

    function compute_area(mesh_body)
        dim = Tesserae.get_dimension(Tesserae.cellshape(mesh_body))
        ParticleProp = @NamedTuple begin
            x      :: Vec{dim, Float64}
            n      :: Vec{dim, Float64}
            detJdA :: Float64
        end
        mesh_face = Tesserae.extract_face(mesh_body, 1:length(mesh_body))
        points = generate_particles(ParticleProp, mesh_face)
        weights = generate_basis_weights(mesh_face, size(points))
        update!(weights, points, mesh_face; normal=points.n, measure=points.detJdA)
        sum(points.detJdA)
    end
    cmesh = CartesianMesh(1, (0,1), (0,1))
    @test compute_area(FEMesh(Tesserae.Quad4(), cmesh)) ≈
          compute_area(FEMesh(Tesserae.Quad8(), cmesh)) ≈
          compute_area(FEMesh(Tesserae.Quad9(), cmesh)) ≈ 4
    @test compute_area(FEMesh(Tesserae.Tri3(), cmesh)) ≈
          compute_area(FEMesh(Tesserae.Tri6(), cmesh)) ≈ 4 + 2√2
    cmesh = CartesianMesh(1, (0,1), (0,1), (0,1))
    @test compute_area(FEMesh(Tesserae.Hex8(), cmesh))  ≈
          compute_area(FEMesh(Tesserae.Hex20(), cmesh)) ≈
          compute_area(FEMesh(Tesserae.Hex27(), cmesh)) ≈ 6
    @test compute_area(FEMesh(Tesserae.Tet4(), cmesh))  ≈
          compute_area(FEMesh(Tesserae.Tet10(), cmesh)) ≈ 6 * (1+√2)

    line_mesh = FEMesh(
        Tesserae.Line2(),
        [Vec(0.0,0.0,0.0), Vec(1.0,2.0,2.0)],
        [Tesserae.SVector(1, 2)],
    )
    LinePointProp = @NamedTuple begin
        x      :: Vec{3, Float64}
        n      :: Vec{3, Float64}
        detJdL :: Float64
    end
    line_points = generate_particles(LinePointProp, line_mesh)
    line_weights = generate_basis_weights(line_mesh, size(line_points))
    update!(line_weights, line_points, line_mesh; measure=line_points.detJdL)
    @test sum(line_points.detJdL) ≈ 3
    @test_throws ArgumentError update!(line_weights, line_points, line_mesh; normal=line_points.n)

    cmesh1 = CartesianMesh(1, (0,2), (0,2))
    cmesh2 = CartesianMesh(1, (1,3), (1,3))
    @test compute_volume(merge(FEMesh(Tesserae.Quad4(), cmesh1), FEMesh(Tesserae.Quad4(), cmesh2))) ≈
          compute_volume(merge(FEMesh(Tesserae.Quad8(), cmesh1), FEMesh(Tesserae.Quad8(), cmesh2))) ≈
          compute_volume(merge(FEMesh(Tesserae.Quad9(), cmesh1), FEMesh(Tesserae.Quad9(), cmesh2))) ≈
          compute_volume(merge(FEMesh(Tesserae.Tri3(), cmesh1), FEMesh(Tesserae.Tri3(), cmesh2)))   ≈
          compute_volume(merge(FEMesh(Tesserae.Tri6(), cmesh1), FEMesh(Tesserae.Tri6(), cmesh2)))   ≈ 7

    cmesh1 = CartesianMesh(1, (0,2), (0,2), (0,2))
    cmesh2 = CartesianMesh(1, (1,3), (1,3), (1,3))
    @test compute_volume(merge(FEMesh(Tesserae.Hex8(), cmesh1), FEMesh(Tesserae.Hex8(), cmesh2)))   ≈
          compute_volume(merge(FEMesh(Tesserae.Hex20(), cmesh1), FEMesh(Tesserae.Hex20(), cmesh2))) ≈
          compute_volume(merge(FEMesh(Tesserae.Hex27(), cmesh1), FEMesh(Tesserae.Hex27(), cmesh2))) ≈
          compute_volume(merge(FEMesh(Tesserae.Tet4(), cmesh1), FEMesh(Tesserae.Tet4(), cmesh2)))   ≈
          compute_volume(merge(FEMesh(Tesserae.Tet10(), cmesh1), FEMesh(Tesserae.Tet10(), cmesh2))) ≈ 15
end
