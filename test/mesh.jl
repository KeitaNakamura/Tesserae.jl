@testset "CartesianMesh" begin
    # constructor
    ## default
    (@inferred CartesianMesh(Float32, 1, (0,3), (1,4)))::CartesianMesh{2, Float32}
    (@inferred CartesianMesh(Float64, 1, (0,3), (1,4), (0,2)))::CartesianMesh{3, Float64}
    mesh = (@inferred CartesianMesh(1, (0,3), (1,4), (0,2)))::CartesianMesh{3, Float64}
    @test mesh[1] === Vec(0.0,1.0,0.0)
    @test mesh[end] === Vec(3.0,4.0,2.0)
    @test mesh == map(Vec, (Iterators.product(range(0,3,step=1), range(1,4,step=1), range(0,2,step=1))))
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

    # isinside
    @test (@inferred Tesserae.isinside(Vec(0.1,0.3), mesh)) === true
    @test (@inferred Tesserae.isinside(Vec(0.2,0.4), mesh)) === true
    ## exactly on the boundary
    @test (@inferred Tesserae.isinside(Vec(0.0,0.0), mesh)) === true
    @test (@inferred Tesserae.isinside(Vec(3.0,4.0), mesh)) === false
    ## outside
    @test (@inferred Tesserae.isinside(Vec(-1.0,3.0), mesh)) === false
    @test (@inferred Tesserae.isinside(Vec(1.0,-3.0), mesh)) === false

    # neighboringnodes
    @test (@inferred neighboringnodes(Vec(0.1,0.1), 1, mesh)) === CartesianIndices((1:2,1:2))
    @test (@inferred neighboringnodes(Vec(0.3,0.1), 2, mesh)) === CartesianIndices((1:4,1:3))
    @test (@inferred neighboringnodes(Vec(0.1,0.3), 2, mesh)) === CartesianIndices((1:3,1:4))
    ## exactly on the node
    @test (@inferred neighboringnodes(Vec(0.2,0.4), 1, mesh)) === CartesianIndices((2:3,3:4))
    @test (@inferred neighboringnodes(Vec(0.2,0.4), 2, mesh)) === CartesianIndices((1:4,2:5))
    @test (@inferred neighboringnodes(Vec(3.0,4.0), 2, mesh)) === CartesianIndices((1:0,1:0))
    ## outside
    @test (@inferred neighboringnodes(Vec(-0.1,3.05), 3, mesh)) === CartesianIndices((1:0,1:0))

    # whichcell
    @test Tesserae.whichcell(Vec(0.1,0.1), mesh) === CartesianIndex(1,1)
    @test Tesserae.whichcell(Vec(2.3,1.1), mesh) === CartesianIndex(12,6)
    ## exactly on the node
    @test Tesserae.whichcell(Vec(0.0,0.0), mesh) === CartesianIndex(1,1)
    @test Tesserae.whichcell(Vec(3.0,4.0), mesh) === nothing
end

@testset "UnstructuredMesh" begin
    cmesh = CartesianMesh(0.5, (0,2), (0,3))
    mesh = UnstructuredMesh(cmesh)
    @test length(mesh) == 35
    @test Tesserae.ncells(mesh) == 24
    @test mesh == vec(cmesh)
    cmesh′ = CartesianMesh(0.5, (1,3), (1,4))
    mesh .= vec(cmesh′) # test setindex!
    @test mesh == vec(cmesh′)
    @test Tesserae.cellshape(UnstructuredMesh(CartesianMesh(1, (0,2)))) == Tesserae.Line2()
    @test Tesserae.cellshape(UnstructuredMesh(CartesianMesh(1, (0,2), (0,3)))) == Tesserae.Quad4()
    @test Tesserae.cellshape(UnstructuredMesh(CartesianMesh(1, (0,2), (0,3), (0,4)))) == Tesserae.Hex8()

    function compute_volume(mesh)
        dim = Tesserae.get_dimension(Tesserae.cellshape(mesh))
        ParticleProp = @NamedTuple begin
            x      :: Vec{dim, Float64}
            detJdV :: Float64
        end
        particles = generate_particles(ParticleProp, mesh)
        weights = generate_interpolation_weights(mesh, size(particles))
        feupdate!(weights, mesh; volume = particles.detJdV)
        sum(particles.detJdV)
    end
    cmesh = CartesianMesh(0.5, (0,2))
    @test compute_volume(UnstructuredMesh(Tesserae.Line2(), cmesh)) ≈
          compute_volume(UnstructuredMesh(Tesserae.Line3(), cmesh)) ≈ 2
    cmesh = CartesianMesh(0.5, (0,2), (-1,3))
    @test compute_volume(UnstructuredMesh(Tesserae.Quad4(), cmesh)) ≈
          compute_volume(UnstructuredMesh(Tesserae.Quad8(), cmesh)) ≈
          compute_volume(UnstructuredMesh(Tesserae.Quad9(), cmesh)) ≈
          compute_volume(UnstructuredMesh(Tesserae.Tri3(), cmesh))  ≈
          compute_volume(UnstructuredMesh(Tesserae.Tri6(), cmesh))  ≈ 8
    cmesh = CartesianMesh(0.5, (0,2), (-1,3), (2,5))
    @test compute_volume(UnstructuredMesh(Tesserae.Hex8(), cmesh))  ≈
          compute_volume(UnstructuredMesh(Tesserae.Hex20(), cmesh)) ≈
          compute_volume(UnstructuredMesh(Tesserae.Hex27(), cmesh)) ≈
          compute_volume(UnstructuredMesh(Tesserae.Tet4(), cmesh))  ≈
          compute_volume(UnstructuredMesh(Tesserae.Tet10(), cmesh)) ≈ 24

    function compute_area(mesh_body)
        dim = Tesserae.get_dimension(Tesserae.cellshape(mesh_body))
        ParticleProp = @NamedTuple begin
            x      :: Vec{dim, Float64}
            n      :: Vec{dim, Float64}
            detJdA :: Float64
        end
        mesh_face = Tesserae.extract_face(mesh_body, 1:length(mesh_body))
        particles = generate_particles(ParticleProp, mesh_face)
        weights = generate_interpolation_weights(mesh_face, size(particles))
        feupdate!(weights, mesh_face; normal = particles.n, area = particles.detJdA)
        sum(particles.detJdA)
    end
    cmesh = CartesianMesh(1, (0,1), (0,1))
    @test compute_area(UnstructuredMesh(Tesserae.Quad4(), cmesh)) ≈
          compute_area(UnstructuredMesh(Tesserae.Quad8(), cmesh)) ≈
          compute_area(UnstructuredMesh(Tesserae.Quad9(), cmesh)) ≈ 4
    @test compute_area(UnstructuredMesh(Tesserae.Tri3(), cmesh)) ≈
          compute_area(UnstructuredMesh(Tesserae.Tri6(), cmesh)) ≈ 4 + 2√2
    cmesh = CartesianMesh(1, (0,1), (0,1), (0,1))
    @test compute_area(UnstructuredMesh(Tesserae.Hex8(), cmesh))  ≈
          compute_area(UnstructuredMesh(Tesserae.Hex20(), cmesh)) ≈
          compute_area(UnstructuredMesh(Tesserae.Hex27(), cmesh)) ≈ 6
    @test compute_area(UnstructuredMesh(Tesserae.Tet4(), cmesh))  ≈
          compute_area(UnstructuredMesh(Tesserae.Tet10(), cmesh)) ≈ 6 * (1+√2)

    cmesh1 = CartesianMesh(1/3, (-1,1), (0,6))
    cmesh2 = CartesianMesh(1/3, (-5,5), (4,5))
    @test compute_volume(merge(UnstructuredMesh(Tesserae.Quad4(), cmesh1), UnstructuredMesh(Tesserae.Quad4(), cmesh2))) ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Quad8(), cmesh1), UnstructuredMesh(Tesserae.Quad8(), cmesh2))) ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Quad9(), cmesh1), UnstructuredMesh(Tesserae.Quad9(), cmesh2))) ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Tri3(), cmesh1), UnstructuredMesh(Tesserae.Tri3(), cmesh2)))   ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Tri6(), cmesh1), UnstructuredMesh(Tesserae.Tri6(), cmesh2)))   ≈ 20

    cmesh1 = CartesianMesh(1/3, (-1,1), (0,6), (-1,1))
    cmesh2 = CartesianMesh(1/3, (-5,5), (4,5), (-5,5))
    @test compute_volume(merge(UnstructuredMesh(Tesserae.Hex8(), cmesh1), UnstructuredMesh(Tesserae.Hex8(), cmesh2)))   ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Hex20(), cmesh1), UnstructuredMesh(Tesserae.Hex20(), cmesh2))) ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Hex27(), cmesh1), UnstructuredMesh(Tesserae.Hex27(), cmesh2))) ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Tet4(), cmesh1), UnstructuredMesh(Tesserae.Tet4(), cmesh2)))   ≈
          compute_volume(merge(UnstructuredMesh(Tesserae.Tet10(), cmesh1), UnstructuredMesh(Tesserae.Tet10(), cmesh2))) ≈ 120
end
