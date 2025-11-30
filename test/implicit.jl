@testset "P2G_Matrix" begin
    interp = BSpline(Quadratic())
    mesh = CartesianMesh(1, (0,10), (0,20))
    grid = generate_grid(@NamedTuple{x::Vec{2,Float64}}, mesh)
    particles = generate_particles(@NamedTuple{x::Vec{2,Float64}}, mesh)
    weights = generate_interpolation_weights(interp, mesh, length(particles))
    update!(weights, particles, mesh)
    A = create_sparse_matrix(interp, mesh)
    B = create_sparse_matrix(interp, mesh)
    @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
        A[i,j] = @∑ w[ip] * sum(∇w[jp])
    end
    @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
        B[j,i] = @∑ w[jp] * sum(∇w[ip])
    end
    @test !(A ≈ A')
    @test !(B ≈ B')
    @test A ≈ B
end

@testset "Backtracking" begin
    @testset "Scalar cubic: backtracking stabilizes" begin
        F(v) = [v[1]^3 - 1e6]
        ∇F(v) = reshape([3v[1]^2], 1, 1)
        x0 = [1.0]

        # without backtracking
        x = copy(x0)
        solved = Tesserae.newton!(x, F, ∇F; rtol=0.0, atol=1e-12, maxiter=10, backtracking=false, verbose=false)
        @test !solved || abs(x[1] - (1e6)^(1/3)) > 1e-3

        # with backtracking
        x = copy(x0)
        solved = Tesserae.newton!(x, F, ∇F; rtol=0.0, atol=1e-12, maxiter=100, backtracking=true, verbose=false)
        @test solved
        @test isapprox(x[1], (1e6)^(1/3); rtol=0.0, atol=1e-12)
    end
end
