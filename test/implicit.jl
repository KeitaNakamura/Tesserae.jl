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
