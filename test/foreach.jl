@testset "Foreach macro" begin
    @testset "Grid" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        GridProp = @NamedTuple{x::Vec{2,Float64}, m::Float64, mv::Vec{2,Float64}, v::Vec{2,Float64}}
        grid = generate_grid(GridProp, mesh)
        @. grid.m = 2
        @. grid.mv = Vec(1, -1)

        @foreach grid=>i begin
            v[i] = x[i] + mv[i] / m[i]
        end

        @test all(i -> grid.v[i] == grid.x[i] + grid.mv[i] / grid.m[i], eachindex(grid))

        @. grid.v = zero(grid.v)
        @threaded @foreach grid=>i begin
            v[i] = x[i] - mv[i] / m[i]
        end

        @test all(i -> grid.v[i] == grid.x[i] - grid.mv[i] / grid.m[i], eachindex(grid))
    end

    @testset "Particles" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        ParticleProp = @NamedTuple{x::Vec{2,Float64}, v::Vec{2,Float64}}
        particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
        @. particles.v = Vec(0.25, -0.5)
        x0 = copy(particles.x)
        Δt = 0.2

        @foreach particles=>p begin
            x[p] += v[p] * Δt
        end

        @test particles.x == x0 .+ particles.v .* Δt
    end

    @testset "Interpolation" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)
        calls = Ref(0)
        scale() = (calls[] += 1; 3.0)

        @foreach grid=>i begin
            m[i] = $(scale()) * (x[i][1] + 1)
        end

        @test calls[] == 1
        @test all(i -> grid.m[i] == 3 * (grid.x[i][1] + 1), eachindex(grid))
    end

    @testset "SpGrid" begin
        mesh = CartesianMesh(1.0, (0,8), (0,8))
        GridProp = @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}
        grid = generate_grid(SpArray, GridProp, mesh)
        particles = [Vec(1.0, 1.0), Vec(7.0, 7.0)]
        update_sparsity!(grid, particles)

        @foreach grid=>i begin
            m[i] = 1
            v[i] = x[i]
        end

        active = collect(Tesserae.activeindices(grid.m))
        @test !isempty(active)
        @test all(i -> grid.m[i] == 1, active)
        @test all(i -> grid.v[i] == grid.x[i], active)
        @test all(i -> iszero(grid.m[i]), filter(i -> !Tesserae.isactive(grid.m, i), eachindex(grid.m)))
        @test all(i -> iszero(grid.v[i]), filter(i -> !Tesserae.isactive(grid.v, i), eachindex(grid.v)))
    end
end
