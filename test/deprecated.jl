@testset "Deprecated" begin
    mesh = CartesianMesh(0.5, (0,2), (0,2))
    grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)
    particles = generate_particles(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)
    weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
    update!(weights, particles, mesh)
    bw = weights[1]
    x = particles.x[1]

    partition = Partition(mesh)
    update!(partition, particles.x)

    femesh = FEMesh(Tesserae.Quad4(), mesh)
    feweights = generate_basis_weights(femesh, 4, Tesserae.ncells(femesh))

    # Deprecated bindings warn outside the logger, so only the identity is testable.
    @test Tesserae.ThreadPartition === Partition
    @test Tesserae.ColorPartition === Partition
    @test Tesserae.Interpolation === Basis
    @test Tesserae.InterpolationWeight === BasisWeight
    @test Tesserae.InterpolationWeightArray === BasisWeightArray

    @test (@test_deprecated Tesserae.whichcell(x, mesh)) == findcell(x, mesh)
    @test (@test_deprecated generate_interpolation_weights(BSpline(Linear()), mesh, length(particles))) isa BasisWeightArray
    @test (@test_deprecated Tesserae.initial_neighboringnodes(BSpline(Linear()), mesh)) == Tesserae.initial_supportnodes(BSpline(Linear()), mesh)
    @test (@test_deprecated Tesserae.neighboringnodes_storage(bw)) === Tesserae.supportnodes_storage(bw)
    @test (@test_deprecated Tesserae.colorgroups(partition)) === threadsafe_groups(partition)
    region = first(first(threadsafe_groups(partition)))
    @test (@test_deprecated Tesserae.particle_indices_in(partition, particles, region)) == particle_indices(partition, particles, region)

    @test (@test_deprecated Tesserae.interpolation(bw)) === basis(bw)
    @test (@test_deprecated Tesserae.interpolation(weights)) === basis(weights)
    @test (@test_deprecated Tesserae.cellshape(feweights[1])) === basis(feweights[1])
    @test (@test_deprecated Tesserae.cellshape(feweights)) === basis(feweights)

    @test (@test_deprecated Tesserae.neighboringnodes(bw)) == supportnodes(bw)
    @test (@test_deprecated Tesserae.neighboringnodes(bw, grid)) == supportnodes(bw, grid)
    @test (@test_deprecated Tesserae.neighboringnodes(BSpline(Linear()), x, mesh)) == supportnodes(BSpline(Linear()), x, mesh)
    @test (@test_deprecated Tesserae.neighboringnodes(x, 1.0, mesh)) == supportnodes(x, 1.0, mesh)
end
