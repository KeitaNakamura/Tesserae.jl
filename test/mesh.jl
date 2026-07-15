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
    @test supportnodes(mesh, 1) === mesh.cellsupports[1]
    @test supportnodes(mesh) === mesh.usednodes
    @test supportnodes(mesh) == collect(eachindex(mesh))
    @test mesh == vec(cmesh)
    cmesh′ = CartesianMesh(0.5, (1,3), (1,4))
    mesh .= vec(cmesh′) # test setindex!
    @test mesh == vec(cmesh′)
    @test Tesserae.cellshape(FEMesh(CartesianMesh(1, (0,2)))) == Tesserae.Line2()
    @test Tesserae.cellshape(FEMesh(CartesianMesh(1, (0,2), (0,3)))) == Tesserae.Quad4()
    @test Tesserae.cellshape(FEMesh(CartesianMesh(1, (0,2), (0,3), (0,4)))) == Tesserae.Hex8()

    mesh_with_unused_node = Tesserae.FEMesh(
        Tesserae.Line2(),
        [Vec(0.0), Vec(1.0), Vec(2.0)],
        [Tesserae.SVector(1, 3)],
    )
    @test length(mesh_with_unused_node) == 3
    @test supportnodes(mesh_with_unused_node) == [1, 3]
    compact_mesh = FEMesh(Tesserae.Line2(), mesh_with_unused_node)
    @test length(compact_mesh) == 2
    @test compact_mesh.cellsupports == [Tesserae.SVector(1, 2)]

    geometry = FEMesh(Tesserae.Quad9(), CartesianMesh(1, (0,2), (0,1)))
    velocity = @inferred FEMesh(Tesserae.Quad9(), geometry)
    pressure = @inferred FEMesh(Tesserae.Quad4(), geometry)
    @test length(geometry) == length(velocity) == 15
    @test length(pressure) == 6
    @test parent(velocity.nodes) === parent(pressure.nodes) === geometry.nodes
    @test pressure.cellsupports == [Tesserae.SVector(1, 2, 5, 4), Tesserae.SVector(2, 3, 6, 5)]
    @test supportnodes(pressure) == collect(eachindex(pressure))
    geometry[1] = Vec(-1.0, -1.0)
    @test velocity[1] == pressure[1] == geometry[1]
    pressure[1] = Vec(-2.0, -2.0)
    @test velocity[1] == geometry[1] == pressure[1]
    @test_throws ArgumentError merge!(pressure, FEMesh(Tesserae.Quad4(), geometry))
    @test_throws ArgumentError FEMesh(Tesserae.Tri3(), geometry)
    @test_throws ArgumentError FEMesh(Tesserae.Line3(), FEMesh(Tesserae.Line4(), [Vec(0.0), Vec(1.0), Vec(1 / 3), Vec(2 / 3)], [Tesserae.SVector(1, 2, 3, 4)]))

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
