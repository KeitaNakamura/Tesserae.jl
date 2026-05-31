@testset "Particles" begin
    function in_domain(x, mesh)
        xmin = Tesserae.get_xmin(mesh)
        xmax = Tesserae.get_xmax(mesh)
        all(d -> xmin[d] ≤ x[d] ≤ xmax[d], 1:length(x))
    end

    function minimum_distance(points)
        dmin = Inf
        for i in 1:length(points)-1, j in i+1:length(points)
            dmin = min(dmin, norm(points[i] - points[j]))
        end
        dmin
    end

    @testset "GridSampling" begin
        mesh = CartesianMesh(1.0, (0,2), (0,1))
        points = Tesserae.generate_points(GridSampling(; spacing=1/2), mesh)
        expected = Vec{2,Float64}[
            Vec(x, y) for x in (0.25, 0.75, 1.25, 1.75) for y in (0.25, 0.75)
        ]

        @test eltype(points) == Vec{2,Float64}
        @test length(points) == length(expected)
        @test Set(points) == Set(expected)
    end

    @testset "PoissonDiskSampling" begin
        mesh2 = CartesianMesh(0.1, (0,1), (0,1))
        alg2 = PoissonDiskSampling(; spacing=1/2, rng=Random.MersenneTwister(1234), threaded=false)
        points2 = Tesserae.generate_points(alg2, mesh2)
        l2 = alg2.spacing * spacing(mesh2)
        domain2 = tuple.(Tuple(Tesserae.get_xmin(mesh2)), Tuple(Tesserae.get_xmax(mesh2)))
        d2 = Tesserae.poisson_disk_sampling_minimum_distance(l2, domain2)

        @test eltype(points2) == Vec{2,Float64}
        @test length(points2) > 1
        @test all(x -> in_domain(x, mesh2), points2)
        @test minimum_distance(points2) ≥ d2 * (1 - sqrt(eps(Float64)))

        threaded_points = Tesserae.generate_points(
            PoissonDiskSampling(; spacing=1/2, rng=Random.MersenneTwister(1234), threaded=true), mesh2)
        @test length(threaded_points) > 1
        @test abs(length(threaded_points) - length(points2)) / length(points2) ≤ 0.15
        @test all(x -> in_domain(x, mesh2), threaded_points)
        @test minimum_distance(threaded_points) ≥ d2 * (1 - sqrt(eps(Float64)))

        mesh3 = CartesianMesh(0.5, (0,1), (0,1), (0,1))
        alg3 = PoissonDiskSampling(; spacing=1/2, rng=Random.MersenneTwister(4321), threaded=false)
        points3 = Tesserae.generate_points(alg3, mesh3)
        l3 = alg3.spacing * spacing(mesh3)
        domain3 = tuple.(Tuple(Tesserae.get_xmin(mesh3)), Tuple(Tesserae.get_xmax(mesh3)))
        d3 = Tesserae.poisson_disk_sampling_minimum_distance(l3, domain3)

        @test eltype(points3) == Vec{3,Float64}
        @test length(points3) > 1
        @test all(x -> in_domain(x, mesh3), points3)
        @test minimum_distance(points3) ≥ d3 * (1 - sqrt(eps(Float64)))

        @test Tesserae.poisson_disk_sampling_minimum_distance(0.1, ((0,1), (0,2))) ≈
              (0.938001956 / 1.42825365^(1/2) + 0.0951151 * (0.1/1 + 0.1/2)) * 0.1
    end

    @testset "generate_particles" begin
        mesh = CartesianMesh(0.5, (0,1), (0,1))
        ParticleProp = @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}
        particles = generate_particles(ParticleProp, mesh; alg=GridSampling())

        @test size(particles) == (16,)
        @test particles.x == Tesserae.generate_points(GridSampling(), mesh)
        @test all(iszero, particles.m)
        @test all(iszero, particles.v)
        @test_throws ErrorException generate_particles(@NamedTuple{x::Vec{2,Float64}, bad::Vector{Float64}},
                                                       mesh; alg=GridSampling())
    end
end
