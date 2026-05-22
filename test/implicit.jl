@testset "P2G_Matrix" begin
    basis = BSpline(Quadratic())
    mesh = CartesianMesh(1, (0,10), (0,20))

    grid = generate_grid(@NamedTuple{x::Vec{2,Float64}}, mesh)
    particles = generate_particles(@NamedTuple{x::Vec{2,Float64}}, mesh)

    weights = generate_basis_weights(basis, mesh, length(particles))
    update!(weights, particles, mesh)

    @testset "square matrix" begin
        A = create_sparse_matrix(basis, mesh)
        B = create_sparse_matrix(basis, mesh)
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
    @testset "multiple matrices" begin
        A = create_sparse_matrix(basis, mesh)
        B = create_sparse_matrix(basis, mesh)
        Aref = create_sparse_matrix(basis, mesh)
        Bref = create_sparse_matrix(basis, mesh)

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ w[ip] * w[jp]
            B[i,j] = @∑ sum(∇w[ip]) * sum(∇w[jp])
        end

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aref[i,j] = @∑ w[ip] * w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Bref[i,j] = @∑ sum(∇w[ip]) * sum(∇w[jp])
        end

        @test A ≈ Aref
        @test B ≈ Bref
    end
    @testset "mixed dof matrices" begin
        Kuu = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        Kup = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        Kpu = create_sparse_matrix(basis, mesh; ndofs=(1, 2))
        Kpp = create_sparse_matrix(basis, mesh; ndofs=(1, 1))
        Kuu_ref = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        Kup_ref = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        Kpu_ref = create_sparse_matrix(basis, mesh; ndofs=(1, 2))
        Kpp_ref = create_sparse_matrix(basis, mesh; ndofs=(1, 1))

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Kuu[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
            Kup[i,j] = @∑ ∇w[ip] * w[jp]
            Kpu[i,j] = @∑ w[ip] * ∇w[jp]'
            Kpp[i,j] = @∑ w[ip] * w[jp]
        end

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Kuu_ref[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Kup_ref[i,j] = @∑ ∇w[ip] * w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Kpu_ref[i,j] = @∑ w[ip] * ∇w[jp]'
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Kpp_ref[i,j] = @∑ w[ip] * w[jp]
        end

        @test Kuu ≈ Kuu_ref
        @test Kup ≈ Kup_ref
        @test Kpu ≈ Kpu_ref
        @test Kpp ≈ Kpp_ref
        @test Kup ≈ Kpu'
    end
    @testset "mixed lhs order" begin
        A = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        B = create_sparse_matrix(basis, mesh; ndofs=(1, 2))
        Aref = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        Bref = create_sparse_matrix(basis, mesh; ndofs=(1, 2))

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ ∇w[ip] * w[jp]
            B[j,i] = @∑ ∇w[ip] * w[jp]
        end

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aref[i,j] = @∑ ∇w[ip] * w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Bref[j,i] = @∑ ∇w[ip] * w[jp]
        end

        @test A ≈ Aref
        @test B ≈ Bref
        @test A ≈ B'
    end
    @testset "duplicate matrix" begin
        ex = quote
            @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                A[i,j] = @∑ w[ip] * w[jp]
                A[i,j] += @∑ sum(∇w[ip]) * sum(∇w[jp])
            end
        end
        @test_throws ErrorException macroexpand(@__MODULE__, ex)
    end
    @testset "rectangular blocks" begin
        n = length(grid)

        Aup = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        Bpu = create_sparse_matrix(basis, mesh; ndofs=(1, 2))

        @test size(Aup) == (2n, n)
        @test size(Bpu) == (n, 2n)

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aup[i,j] = @∑ ∇w[ip] * w[jp]
        end

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Bpu[j,i] = @∑ ∇w[ip] * w[jp]
        end

        @test Aup ≈ Bpu'
    end
    @testset "assignment operators" begin
        Aterm = create_sparse_matrix(basis, mesh; ndofs=1)
        Aeq = create_sparse_matrix(basis, mesh; ndofs=1)
        Aplus = create_sparse_matrix(basis, mesh; ndofs=1)
        Aminus = create_sparse_matrix(basis, mesh; ndofs=1)

        fill!(Tesserae.SparseArrays.nonzeros(Aeq), 7)
        fill!(Tesserae.SparseArrays.nonzeros(Aplus), 7)
        fill!(Tesserae.SparseArrays.nonzeros(Aminus), 7)
        Abase = copy(Aplus)

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aterm[i,j] = @∑ w[ip] * w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aeq[i,j] = @∑ w[ip] * w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aplus[i,j] += @∑ w[ip] * w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aminus[i,j] -= @∑ w[ip] * w[jp]
        end

        @test Aeq ≈ Aterm
        @test Aplus ≈ Abase + Aterm
        @test Aminus ≈ Abase - Aterm
    end
    @testset "allocation sanity" begin
        function assemble_mixed_block!(Kuu, Kup, Kpu, Kpp, grid, particles, weights)
            @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                Kuu[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
                Kup[i,j] = @∑ ∇w[ip] * w[jp]
                Kpu[j,i] = @∑ ∇w[ip] * w[jp]
                Kpp[i,j] = @∑ w[ip] * w[jp]
            end
            nothing
        end
        function assemble_mixed_separate!(Kuu, Kup, Kpu, Kpp, grid, particles, weights)
            @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                Kuu[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
            end
            @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                Kup[i,j] = @∑ ∇w[ip] * w[jp]
            end
            @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                Kpu[j,i] = @∑ ∇w[ip] * w[jp]
            end
            @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                Kpp[i,j] = @∑ w[ip] * w[jp]
            end
            nothing
        end
        make_matrices() = (
            create_sparse_matrix(basis, mesh; ndofs=(2, 2)),
            create_sparse_matrix(basis, mesh; ndofs=(2, 1)),
            create_sparse_matrix(basis, mesh; ndofs=(1, 2)),
            create_sparse_matrix(basis, mesh; ndofs=(1, 1)),
        )

        Ks = make_matrices()
        Ks_ref = make_matrices()
        assemble_mixed_block!(Ks..., grid, particles, weights)
        assemble_mixed_separate!(Ks_ref..., grid, particles, weights)

        alloc_block = @allocated assemble_mixed_block!(Ks..., grid, particles, weights)
        alloc_separate = @allocated assemble_mixed_separate!(Ks_ref..., grid, particles, weights)
        @test alloc_block <= 2alloc_separate
    end
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
