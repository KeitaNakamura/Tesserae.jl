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

    lims = ((0,2), (-1,3), (2,5))
    for dim in 1:3
        if dim == 1
            shapes = (Tesserae.Line2(), Tesserae.Line3())
        elseif dim == 2
            shapes = (Tesserae.Quad4(), Tesserae.Quad8(), Tesserae.Quad9(), Tesserae.Tri3(), Tesserae.Tri6())
        elseif dim == 3
            shapes = (Tesserae.Hex8(), Tesserae.Hex20(), Tesserae.Hex27(), Tesserae.Tet4(), Tesserae.Tet10())
        end
        for shape in shapes
            ParticleProp = @NamedTuple begin
                x      :: Vec{dim, Float64}
                detJdV :: Float64
            end
            cmesh = CartesianMesh(0.5, lims[1:dim]...)
            mesh = UnstructuredMesh(shape, cmesh)
            particles = generate_particles(ParticleProp, mesh)
            mpvalues = generate_mpvalues(mesh, size(particles))
            feupdate!(mpvalues, mesh; volume = particles.detJdV)
            @test sum(particles.detJdV) ≈ volume(cmesh)
        end
    end

    lims = ((0,1), (0,1), (0,1))
    function compute_area(shape)
        dim = Tesserae.get_dimension(shape)
        ParticleProp = @NamedTuple begin
            x      :: Vec{dim, Float64}
            n      :: Vec{dim, Float64}
            detJdA :: Float64
        end
        mesh_body = UnstructuredMesh(shape, CartesianMesh(1, lims[1:dim]...))
        mesh_face = Tesserae.extract_face(mesh_body, 1:length(mesh_body))
        particles = generate_particles(ParticleProp, mesh_face)
        mpvalues = generate_mpvalues(mesh_face, size(particles))
        feupdate!(mpvalues, mesh_face; normal = particles.n, area = particles.detJdA)
        sum(particles.detJdA)
    end
    @test compute_area(Tesserae.Quad4()) ≈ compute_area(Tesserae.Quad8()) ≈ compute_area(Tesserae.Quad9()) ≈ 4
    @test compute_area(Tesserae.Hex8()) ≈ compute_area(Tesserae.Hex20()) ≈ compute_area(Tesserae.Hex27()) ≈ 6
    @test compute_area(Tesserae.Tri3()) ≈ compute_area(Tesserae.Tri6()) ≈ 4 + 2√2
    @test compute_area(Tesserae.Tet4()) ≈ compute_area(Tesserae.Tet10()) ≈ 6 * (1+√2)
end
