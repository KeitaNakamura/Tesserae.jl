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

    @testset "Grid slices" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2), (0,2))
        grid = generate_grid(@NamedTuple{x::Vec{3,Float64}, m::Float64}, mesh)
        indices = CartesianIndices(size(grid))

        fillzero!(grid.m)
        @foreach grid[:,:,begin]=>i begin
            m[i] = 1
        end

        @test all(i -> grid.m[i] == (i[3] == 1 ? 1 : 0), indices)

        fillzero!(grid.m)
        @threaded @foreach grid[end,:,:]=>i begin
            m[i] = 2
        end

        @test all(i -> grid.m[i] == (i[1] == size(grid, 1) ? 2 : 0), indices)

        fillzero!(grid.m)
        @foreach grid[begin+1:end-1,:,:]=>i begin
            m[i] = 3
        end

        @test all(i -> grid.m[i] == (1 < i[1] < size(grid, 1) ? 3 : 0), indices)

        fillzero!(grid.m)
        @foreach grid[begin:2:end,:,end]=>i begin
            m[i] = 4
        end

        @test all(i -> grid.m[i] == (isodd(i[1]) && i[3] == size(grid, 3) ? 4 : 0), indices)

        fillzero!(grid.m)
        @foreach grid[begin,end,begin]=>i begin
            m[i] = 5
        end

        fixed_index = CartesianIndex(1, size(grid, 2), 1)
        @test all(i -> grid.m[i] == (i == fixed_index ? 5 : 0), indices)
    end

    @testset "Grid empty slices" begin
        mesh = CartesianMesh(1.0, (0,1), (0,1))
        grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)

        @foreach grid[begin+1:end-1,:]=>i begin
            m[i] = 1
        end

        @test all(iszero, grid.m)
    end

    @testset "Grid slice bounds" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)

        @test_throws BoundsError @foreach grid[begin-1,:]=>i begin
            m[i] = 1
        end

        @test_throws BoundsError @foreach grid[:,end+1]=>i begin
            m[i] = 1
        end
    end

    @testset "Grid slice selectors" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)

        @test_throws ArgumentError @foreach grid[[1,2],:]=>i begin
            m[i] = 1
        end
    end

    @testset "Grid slice dimensions" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)

        @test_throws ArgumentError @foreach grid[:,:,begin]=>i begin
            m[i] = 1
        end
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

    @testset "Particle slices" begin
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        particles = generate_particles(@NamedTuple{x::Vec{2,Float64}, v::Vec{2,Float64}}, mesh; alg=GridSampling())
        @. particles.v = Vec(0.25, -0.5)

        @foreach particles[begin:end]=>p begin
            v[p] = -v[p]
        end

        @test particles.v == fill(Vec(-0.25, 0.5), length(particles))

        @threaded @foreach particles[begin+1:end-1]=>p begin
            v[p] = Vec(1, -1)
        end

        @test particles.v[begin] == Vec(-0.25, 0.5)
        @test particles.v[end] == Vec(-0.25, 0.5)
        @test all(p -> particles.v[p] == Vec(1, -1), 2:(length(particles)-1))
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

        fillzero!(grid.m)
        @foreach grid[:,end]=>i begin
            m[i] = $(scale()) * (x[i][2] + 1)
        end

        @test calls[] == 2
        @test all(eachindex(grid)) do i
            expected = i[2] == size(grid, 2) ? 3 * (grid.x[i][2] + 1) : 0
            grid.m[i] == expected
        end
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

    @testset "SpGrid slices" begin
        mesh = CartesianMesh(1.0, (0,8), (0,8))
        grid = generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, mesh)
        update_sparsity!(grid, trues(Tesserae.nblocks(mesh)))

        @. grid.m = 0
        @foreach grid[:,begin]=>i begin
            m[i] = 2
            v[i] = x[i]
        end

        active = collect(Tesserae.activeindices(grid.m))
        @test !isempty(active)
        @test all(active) do i
            I = Tesserae.logicalindex(i)
            if I[2] == 1
                grid.m[i] == 2 && grid.v[i] == grid.x[i]
            else
                iszero(grid.m[i]) && iszero(grid.v[i])
            end
        end

        fillzero!(grid.m)
        @threaded @foreach grid[end,:]=>i begin
            m[i] = 3
        end

        @test all(active) do i
            I = Tesserae.logicalindex(i)
            I[1] == size(grid, 1) ? grid.m[i] == 3 : iszero(grid.m[i])
        end

        sparse_grid = generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh)
        update_sparsity!(sparse_grid, [Vec(1.0, 1.0), Vec(7.0, 7.0)])

        @foreach sparse_grid[:,begin]=>i begin
            m[i] = 4
        end

        sparse_active = collect(Tesserae.activeindices(sparse_grid.m))
        @test any(i -> Tesserae.logicalindex(i)[2] == 1, sparse_active)
        @test all(sparse_active) do i
            I = Tesserae.logicalindex(i)
            I[2] == 1 ? sparse_grid.m[i] == 4 : iszero(sparse_grid.m[i])
        end

        sparse_inactive = filter(i -> !Tesserae.isactive(sparse_grid.m, i), eachindex(sparse_grid.m))
        @test all(i -> iszero(sparse_grid.m[i]), sparse_inactive)
    end
end
