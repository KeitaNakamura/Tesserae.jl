@testset "P2G_Matrix" begin
    basis = BSpline(Quadratic())
    mesh = CartesianMesh(1, (0,4), (0,5))

    grid = generate_grid(@NamedTuple{x::Vec{2,Float64}}, mesh)
    particles = generate_particles(@NamedTuple{x::Vec{2,Float64}}, mesh; alg=GridSampling())

    weights = generate_basis_weights(basis, mesh, length(particles))
    update!(weights, particles, mesh)

    @test_throws UndefKeywordError create_sparse_matrix(basis, mesh)

    @testset "square matrix" begin
        A = create_sparse_matrix(basis, mesh; ndofs=2)
        B = create_sparse_matrix(basis, mesh; ndofs=2)
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
        A = create_sparse_matrix(basis, mesh; ndofs=2)
        B = create_sparse_matrix(basis, mesh; ndofs=2)
        Aref = create_sparse_matrix(basis, mesh; ndofs=2)
        Bref = create_sparse_matrix(basis, mesh; ndofs=2)

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
    @testset "matrix assembly helpers" begin
        A = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        B = create_sparse_matrix(basis, mesh; ndofs=(1, 2))

        table_i, table_j = Tesserae.matrix_dof_tables(A, grid, grid)
        @test size(table_i) == (2, size(grid)...)
        @test size(table_j) == (1, size(grid)...)
        @test size(A) == (length(table_i), length(table_j))

        table_j_transposed, table_i_transposed = Tesserae.matrix_dof_tables(B, grid, grid)
        @test size(table_i_transposed) == size(table_i)
        @test size(table_j_transposed) == size(table_j)
        @test size(B) == (length(table_j_transposed), length(table_i_transposed))

        bw = first(weights)
        nodes_i, nodes_j = Tesserae.matrix_supportnodes(bw, grid)
        @test nodes_i === nodes_j
        @test Tesserae.matrix_supportnodes(bw, grid, bw, grid) == (nodes_i, nodes_j)

        local_i = Tesserae.local_dof_table(table_i, nodes_i)
        local_j = Tesserae.local_dof_table(table_j, nodes_j)
        ip = first(eachindex(nodes_i))
        jp = first(eachindex(nodes_j))
        @test Tesserae.local_dofs(local_i, ip) == vec(view(local_i, :, ip))
        @test Tesserae.local_dofs(local_j, jp) == vec(view(local_j, :, jp))

        dofs_i, dofs_j = Tesserae.support_dofs(table_i, nodes_i, table_j, nodes_j)
        @test dofs_i == vec(table_i[:, nodes_i])
        @test dofs_j == vec(table_j[:, nodes_j])

        scalar_table_i, scalar_table_j = Tesserae.matrix_dof_tables(create_sparse_matrix(basis, mesh; ndofs=1), grid, grid)
        scalar_dofs_i, scalar_dofs_j = Tesserae.support_dofs(scalar_table_i, nodes_i, scalar_table_j, nodes_j)
        @test scalar_dofs_i === scalar_dofs_j
    end
end

@testset "Newton" begin
    @testset "Initial nonfinite residual preserves x" begin
        f(v) = fill(Inf, length(v))
        J(v) = Matrix{Float64}(I, length(v), length(v))

        x = [1.0, 2.0, 4.0, 8.0]
        x0 = copy(x)
        solved = Tesserae.newton!(x, f, J; verbose=false)

        @test !solved
        @test x == x0
    end

    @testset "Maxiter preserves finite last iterate" begin
        f(v) = [v[1]^2 - 2]
        J(v) = reshape([2v[1]], 1, 1)

        x = [2.0]
        solved = Tesserae.newton!(x, f, J; rtol=0.0, atol=0.0, maxiter=1, verbose=false)

        @test !solved
        @test x == [1.5]
    end
end

@testset "Backtracking" begin
    @testset "Rejects non-descent direction before trial" begin
        visits = Float64[]
        f(v) = (push!(visits, v[1]); [v[1] - 1])
        J(v) = reshape([1.0], 1, 1)
        linsolve(δx, A, b) = (δx .= -b; δx)

        x = [0.0]
        solved = Tesserae.newton!(x, f, J; linsolve=linsolve, backtracking=true, verbose=false)

        @test !solved
        @test x == [0.0]
        @test visits == [0.0]
    end

    @testset "Failed line search restores accepted state" begin
        visits = Float64[]
        state = Ref(NaN)
        f(v) = (state[] = v[1]; push!(visits, v[1]); [1.0])
        J(v) = reshape([1.0], 1, 1)

        x = [2.0]
        solved = Tesserae.newton!(x, f, J; backtracking=true, verbose=false)

        @test !solved
        @test x == [2.0]
        @test state[] == 2.0
        @test visits[1] == 2.0
        @test visits[end] == 2.0
        @test length(visits) > 2
        @test any(v -> v != 2.0, visits)
    end

    @testset "Scalar cubic: backtracking stabilizes" begin
        f(v) = [v[1]^3 - 1e6]
        J(v) = reshape([3v[1]^2], 1, 1)
        x0 = [1.0]

        # without backtracking
        x = copy(x0)
        solved = Tesserae.newton!(x, f, J; rtol=0.0, atol=1e-12, maxiter=10, backtracking=false, verbose=false)
        @test !solved || abs(x[1] - (1e6)^(1/3)) > 1e-3

        # with backtracking
        x = copy(x0)
        solved = Tesserae.newton!(x, f, J; rtol=0.0, atol=1e-12, maxiter=100, backtracking=true, verbose=false)
        @test solved
        @test isapprox(x[1], (1e6)^(1/3); rtol=0.0, atol=1e-12)
    end
end

@testset "DofMap and sparse extraction" begin
    mesh = CartesianMesh(1, (0,2), (0,1))
    grid = generate_grid(@NamedTuple{x::Vec{2,Float64}, u::Float64, s::Vec{1,Float64}, v::Vec{2,Float64}}, mesh)

    grid.u .= reshape(1.0:length(grid), size(grid))
    grid.s .= map(x -> Vec(x), grid.u)
    grid.v .= reshape(reinterpret(Vec{2,Float64}, 1.0:2length(grid)), size(grid))

    vmask = falses(2, size(grid)...)
    vmask[1, 1:2, :] .= true
    vmask[:, 3, 2] .= true
    vmap = DofMap(vmask)

    @test ndofs(vmap) == count(vmask)
    @test collect(vmap(grid.v)) == [1.0, 3.0, 7.0, 9.0, 11.0, 12.0]
    vmap(grid.v) .= -1:-1:-ndofs(vmap)
    @test collect(vmap(grid.v)) == collect(-1.0:-1.0:-Float64(ndofs(vmap)))

    smask = falses(1, size(grid)...)
    smask[1, 1, 1] = true
    smask[1, 3, 2] = true
    smap = DofMap(smask)

    @test collect(smap(grid.u)) == [1.0, 6.0]
    @test collect(smap(grid.s)) == [1.0, 6.0]

    A = reshape(1.0:36.0, 6, 6)
    @test extract(A, smap) == A[Tesserae.dofs(smap), Tesserae.dofs(smap)]
    @test extract(A, :, smap) == A[:, Tesserae.dofs(smap)]
    @test extract(view, A, smap, :) == view(A, Tesserae.dofs(smap), :)
end

@testset "FEM sparse matrix pattern" begin
    cmesh = CartesianMesh(1, (0,1), (0,1))
    geometry = FEMesh(Tesserae.Quad9(), cmesh)
    quad4 = FEMesh(Tesserae.Quad4(), geometry)
    quad9 = FEMesh(Tesserae.Quad9(), geometry)

    @test_throws UndefKeywordError create_sparse_matrix(quad4)

    A = create_sparse_matrix((quad9, quad4); ndofs=(2, 1))
    @test size(A) == (18, 4)
    @test Tesserae.SparseArrays.nnz(A) == prod(size(A))

    shifted = FEMesh(Tesserae.Quad4(), CartesianMesh(1, (2,3), (2,3)))
    B = create_sparse_matrix((quad9, shifted); ndofs=(2, 1))
    @test size(B) == (2length(quad9), length(shifted))
    @test iszero(Tesserae.SparseArrays.nnz(B))
end
