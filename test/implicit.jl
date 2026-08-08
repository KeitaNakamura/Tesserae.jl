@testset "P2G_Matrix" begin
    basis = BSpline(Quadratic())
    mesh = CartesianMesh(1, (0,4), (0,5))

    grid = generate_grid(@NamedTuple{x::Vec{2,Float64}}, mesh)
    particles = generate_particles(@NamedTuple{x::Vec{2,Float64}}, mesh; alg=GridSampling())

    weights = generate_basis_weights(basis, mesh, length(particles))
    update!(weights, particles, mesh)

    @test_throws UndefKeywordError create_sparse_matrix(basis, mesh)

    @testset "square matrix" begin
        entry = Mat{2,2}(1.0, 2.0, 2.0, 3.0)
        A = create_sparse_matrix(basis, mesh; ndofs=2)
        B = create_sparse_matrix(basis, mesh; ndofs=2)
        A_dense = zeros(size(A))
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ w[ip] * sum(∇w[jp]) * entry
            A_dense[i,j] = @∑ w[ip] * sum(∇w[jp]) * entry
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            B[j,i] = @∑ w[jp] * sum(∇w[ip]) * entry
        end
        @test !(A ≈ A')
        @test !(B ≈ B')
        @test A ≈ B
        @test A ≈ A_dense
    end
    @testset "multiple matrices" begin
        entry = Mat{2,2}(1.0, 2.0, 2.0, 3.0)
        A = create_sparse_matrix(basis, mesh; ndofs=2)
        B = create_sparse_matrix(basis, mesh; ndofs=2)
        Aref = create_sparse_matrix(basis, mesh; ndofs=2)
        Bref = create_sparse_matrix(basis, mesh; ndofs=2)

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ w[ip] * w[jp] * entry
            B[i,j] = @∑ sum(∇w[ip]) * sum(∇w[jp]) * entry
        end

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Aref[i,j] = @∑ w[ip] * w[jp] * entry
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Bref[i,j] = @∑ sum(∇w[ip]) * sum(∇w[jp]) * entry
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
    @testset "block sparse matrix" begin
        Kuu_ref = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        Kup_ref = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        Kpu_ref = create_sparse_matrix(basis, mesh; ndofs=(1, 2))
        Kpp_ref = create_sparse_matrix(basis, mesh; ndofs=(1, 1))
        blocks = @inferred create_block_sparse_matrix(basis, mesh; ndofs=(2, 1))
        blocks32 = @inferred create_block_sparse_matrix(Float32, basis, mesh; ndofs=(2, 1))
        three_fields = @inferred create_block_sparse_matrix(basis, mesh; ndofs=(2, 1, 1))

        @test size(blocks) == (2, 2)
        @test isconcretetype(eltype(blocks))
        @test size(parent(blocks)) == (3length(mesh), 3length(mesh))
        @test eltype(parent(blocks32)) == Float32
        @test size(three_fields) == (3, 3)
        @test size(parent(three_fields)) == (4length(mesh), 4length(mesh))
        @test all(parent(block) === parent(blocks) for block in blocks)
        @test_throws BoundsError blocks[0,1]
        @test_throws BoundsError blocks[1,1][0,1]
        @test_throws BoundsError blocks[1,1][0,1] = 0
        @test size(blocks[1,1]) == size(Kuu_ref)
        @test size(blocks[1,2]) == size(Kup_ref)
        @test size(blocks[2,1]) == size(Kpu_ref)
        @test size(blocks[2,2]) == size(Kpp_ref)
        @test blocks[1,1] == Kuu_ref
        @test blocks[1,2] == Kup_ref
        @test blocks[2,1] == Kpu_ref
        @test blocks[2,2] == Kpp_ref
        @test_throws ArgumentError create_block_sparse_matrix(basis, mesh; ndofs=())
        @test_throws ArgumentError create_block_sparse_matrix(basis, mesh; ndofs=(2, 0))

        assembler = @inferred Tesserae.matrix_assembler(blocks[1,1], mesh, mesh, basis, basis)
        @test Tesserae.has_cartesian_sparse_pattern(assembler)
        linear_basis = BSpline(Linear())
        @test_throws ArgumentError Tesserae.matrix_assembler(blocks[1,1], mesh, mesh, linear_basis, linear_basis)
        different_mesh = CartesianMesh(1, (0,3), (0,5))
        @test_throws DimensionMismatch Tesserae.matrix_assembler(blocks[1,1], different_mesh, different_mesh, basis, basis)

        blocks[1,1][1,1] = 1
        @test parent(blocks)[1,1] == 1
        rowvals_before = copy(Tesserae.SparseArrays.rowvals(parent(blocks)))
        @test_throws ArgumentError blocks[1,1][1,end] = 1
        @test Tesserae.SparseArrays.rowvals(parent(blocks)) == rowvals_before
        fill!(Tesserae.SparseArrays.nonzeros(parent(blocks)), 7)
        unchanged_block = copy(blocks[2,1])
        fillzero!(blocks[1,1])
        @test all(iszero, blocks[1,1])
        @test blocks[2,1] == unchanged_block
        fill!(Tesserae.SparseArrays.nonzeros(parent(blocks)), 7)

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            blocks[1,1][i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
            blocks[1,2][i,j] = @∑ ∇w[ip] * w[jp]
            blocks[2,1][i,j] = @∑ w[ip] * ∇w[jp]'
            blocks[2,2][i,j] = @∑ w[ip] * w[jp]
        end
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            Kuu_ref[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
            Kup_ref[i,j] = @∑ ∇w[ip] * w[jp]
            Kpu_ref[i,j] = @∑ w[ip] * ∇w[jp]'
            Kpp_ref[i,j] = @∑ w[ip] * w[jp]
        end

        @test blocks[1,1] ≈ Kuu_ref
        @test blocks[1,2] ≈ Kup_ref
        @test blocks[2,1] ≈ Kpu_ref
        @test blocks[2,2] ≈ Kpp_ref

        threaded = create_block_sparse_matrix(basis, mesh; ndofs=(2, 1))
        fillzero!(threaded)
        partition = ThreadPartition(mesh)
        update!(partition, particles.x)
        @threaded :static @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
            threaded[1,1][i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
            threaded[1,2][i,j] = @∑ ∇w[ip] * w[jp]
            threaded[2,1][i,j] = @∑ w[ip] * ∇w[jp]'
            threaded[2,2][i,j] = @∑ w[ip] * w[jp]
        end
        @test parent(threaded) ≈ parent(blocks)

        fill!(Tesserae.SparseArrays.nonzeros(parent(blocks)), 7)
        before = copy(parent(blocks))
        @test_throws ArgumentError begin
            @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                blocks[i,j] = @∑ w[ip] * w[jp]
            end
        end
        @test parent(blocks) == before
    end
    @testset "sparse matrix views" begin
        parent_dofs = LinearIndices((3, length(mesh)))
        velocity_dofs = vec(parent_dofs[[3,1], :])
        pressure_dofs = vec(parent_dofs[2:2, :])

        zeroed_parent = create_sparse_matrix(basis, mesh; ndofs=3)
        fill!(Tesserae.SparseArrays.nonzeros(zeroed_parent), 7)
        expected_zeroed_parent = copy(zeroed_parent)
        fill!(view(expected_zeroed_parent, velocity_dofs, pressure_dofs), 0)
        Tesserae.fillzero!(view(zeroed_parent, velocity_dofs, pressure_dofs))
        @test zeroed_parent == expected_zeroed_parent

        parent_sequential = create_sparse_matrix(basis, mesh; ndofs=3)
        parent_threaded = create_sparse_matrix(basis, mesh; ndofs=3)
        parent_transposed = create_sparse_matrix(basis, mesh; ndofs=3)
        sequential = view(parent_sequential, velocity_dofs, pressure_dofs)
        threaded = view(parent_threaded, velocity_dofs, pressure_dofs)
        transposed = view(parent_transposed, pressure_dofs, velocity_dofs)
        reference = create_sparse_matrix(basis, mesh; ndofs=(2, 1))

        assembler = @inferred Tesserae.matrix_assembler(sequential, mesh, mesh, basis, basis)
        @test assembler isa Tesserae.CartesianSparseMatrixAssembler
        @test assembler.matrix === sequential
        @test Tesserae.matrix_storage(assembler.matrix) === parent_sequential
        @test Tesserae.has_cartesian_sparse_pattern(assembler)

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            sequential[i,j] = @∑ ∇w[ip] * w[jp]
            transposed[j,i] = @∑ ∇w[ip] * w[jp]
            reference[i,j] = @∑ ∇w[ip] * w[jp]
        end

        partition = ThreadPartition(mesh)
        update!(partition, particles.x)
        @threaded :static @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
            threaded[i,j] = @∑ ∇w[ip] * w[jp]
        end

        @test sequential ≈ reference
        @test threaded ≈ reference
        @test transposed ≈ reference'
        @test all(iszero, parent_sequential[pressure_dofs, :])
        @test all(iszero, parent_sequential[:, velocity_dofs])

        inconsistent_dofs = copy(velocity_dofs)
        inconsistent_dofs[3] = parent_dofs[2,2]
        inconsistent = view(parent_sequential, inconsistent_dofs, pressure_dofs)
        @test_throws ArgumentError Tesserae.matrix_assembler(inconsistent, mesh, mesh, basis, basis)

        duplicate = view(parent_sequential, vec(parent_dofs[[3,3], :]), pressure_dofs)
        @test_throws ArgumentError Tesserae.matrix_assembler(duplicate, mesh, mesh, basis, basis)
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
    @testset "block partition" begin
        partition = ThreadPartition(mesh)
        update!(partition, particles.x)

        reference = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        block_sparse = create_sparse_matrix(basis, mesh; ndofs=(2, 1))
        block_dense = zeros(length(grid), 2length(grid))

        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            reference[i,j] = @∑ ∇w[ip] * w[jp]
        end
        @threaded :static @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
            block_sparse[i,j] = @∑ ∇w[ip] * w[jp]
            block_dense[j,i] = @∑ ∇w[ip] * w[jp]
        end

        @test block_sparse ≈ reference
        @test block_dense ≈ Matrix(reference')
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
        assembler = @inferred Tesserae.matrix_assembler(A, mesh, mesh, basis, basis)
        @test assembler.matrix === A
        @test assembler.row_dof_table == table_i
        @test assembler.col_dof_table == table_j
        dense = Matrix(A)
        generic_assembler = @inferred Tesserae.matrix_assembler(dense, mesh, mesh, basis, basis)
        @test generic_assembler.matrix === dense
        @test generic_assembler.row_dof_table == table_i
        @test generic_assembler.col_dof_table == table_j
        invalid = Tesserae.SparseArrays.dropzeros!(copy(A))
        @test_throws ArgumentError Tesserae.matrix_assembler(invalid, mesh, mesh, basis, basis)

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

        particle_indices = firstindex(particles):(firstindex(particles) + 2)
        particle_nodes = map(p -> supportnodes(weights[p]), particle_indices)
        first_node = CartesianIndex(map(min, Tuple.(first.(particle_nodes))...))
        last_node = CartesianIndex(map(max, Tuple.(last.(particle_nodes))...))
        @test Tesserae.matrix_block_supportnodes(weights, particle_indices, grid) ==
              first_node:last_node

        ip = length(nodes_i)
        jp = length(nodes_j)
        @test Tesserae.local_dofs(2, ip) == (2ip-1):2ip
        @test Tesserae.local_dofs(1, jp) == jp:jp

        dofs_i, dofs_j = Tesserae.support_dofs(table_i, nodes_i, table_j, nodes_j)
        @test dofs_i == vec(table_i[:, nodes_i])
        @test dofs_j == vec(table_j[:, nodes_j])

        scalar_table_i, scalar_table_j = Tesserae.matrix_dof_tables(create_sparse_matrix(basis, mesh; ndofs=1), grid, grid)
        scalar_dofs_i, scalar_dofs_j = Tesserae.support_dofs(scalar_table_i, nodes_i, scalar_table_j, nodes_j)
        @test scalar_dofs_i === scalar_dofs_j
    end
    @testset "matrix entry sizes" begin
        @test Tesserae.check_matrix_entry_size(1.0, 1, 1)
        @test_throws DimensionMismatch Tesserae.check_matrix_entry_size(1.0, 2, 1)
        @test Tesserae.check_matrix_entry_size(zeros(2), 2, 1)
        @test Tesserae.check_matrix_entry_size(zeros(2), 1, 2)
        @test_throws DimensionMismatch Tesserae.check_matrix_entry_size(zeros(2), 2, 2)
        @test Tesserae.check_matrix_entry_size(zeros(2, 3), 2, 3)
        @test_throws DimensionMismatch Tesserae.check_matrix_entry_size(zeros(2, 2), 2, 3)
    end
    @testset "Block matrix buffer pool" begin
        scatter_mesh = CartesianMesh(1, (0, 4))
        matrix = create_sparse_matrix(basis, scatter_mesh; ndofs=(2, 3))
        assembler = Tesserae.matrix_assembler(matrix, scatter_mesh, scatter_mesh, basis, basis)
        nodes = CartesianIndices((1:3,))
        buffer = Tesserae.BlockMatrixBuffer(Tesserae.BlockMatrixBufferKey(assembler, nodes, nodes))
        pool = Tesserae.BlockMatrixBufferPool()

        fill!(buffer.values, 1)
        @test Tesserae.release!(pool, buffer) === nothing
        reused_buffer = Tesserae.acquire!(pool, buffer.key)
        @test reused_buffer === buffer
        @test all(iszero, reused_buffer.values)
    end
    @testset "Cartesian sparse scatter" begin
        for dims in ((5,), (5, 6, 7))
            scatter_mesh = CartesianMesh(1, ((0, n - 1) for n in dims)...)
            nodes = CartesianIndices(ntuple(_ -> 1:3, length(dims)))
            tail_ranges = ntuple(_ -> 1:3, length(dims) - 1)
            shifted_row_nodes = CartesianIndices((1:2, tail_ranges...))
            shifted_col_nodes = CartesianIndices((2:3, tail_ranges...))
            row_dofs = LinearIndices((2, dims...))
            col_dofs = LinearIndices((3, dims...))

            for (row_nodes, col_nodes) in ((nodes, nodes), (shifted_row_nodes, shifted_col_nodes))
                direct = create_sparse_matrix(basis, scatter_mesh; ndofs=(2, 3))
                merge = copy(direct)
                block_matrix = copy(direct)
                row_size = 2length(row_nodes)
                col_size = 3length(col_nodes)
                local_matrix = reshape(collect(1.0:row_size*col_size), row_size, col_size)

                assembler = Tesserae.matrix_assembler(direct, scatter_mesh, scatter_mesh, basis, basis)
                buffer = Tesserae.BlockMatrixBuffer(Tesserae.BlockMatrixBufferKey(assembler, row_nodes, col_nodes))
                for (jp, col_node) in enumerate(col_nodes), (ip, row_node) in enumerate(row_nodes)
                    I = (2ip-1):2ip
                    J = (3jp-2):3jp
                    Tesserae.add_entry!(buffer, row_nodes, col_nodes, row_node, col_node, @view(local_matrix[I,J]))
                end
                block_assembler = Tesserae.matrix_assembler(block_matrix, scatter_mesh, scatter_mesh, basis, basis)
                @test Tesserae.scatter!(block_assembler, buffer, row_nodes, col_nodes) === block_matrix
                for (jp, col_node) in enumerate(col_nodes), (ip, row_node) in enumerate(row_nodes)
                    I = (2ip-1):2ip
                    J = (3jp-2):3jp
                    Tesserae.add_entry!(assembler, row_node, col_node, @view(local_matrix[I,J]))
                end
                Tesserae.add!(merge, vec(row_dofs[:, row_nodes]), vec(col_dofs[:, col_nodes]), local_matrix)
                @test direct == merge
                @test block_matrix == merge
            end
        end
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
    @testset "block DoF map" begin
        @test Tesserae.dofs(@inferred(dofmap(vmask))) == Tesserae.dofs(vmap)

        blockmap = @inferred dofmap((vmask, smask))
        expected_dofs = vcat(Tesserae.dofs(vmap), 2length(mesh) .+ Tesserae.dofs(smap))
        @test length(blockmap) == 2
        @test ndofs(blockmap) == ndofs(vmap) + ndofs(smap)
        @test Tesserae.dofs(blockmap) == expected_dofs
        @test Tesserae.dofs(blockmap[1]) == Tesserae.dofs(vmap)
        @test Tesserae.dofs(blockmap[2]) == Tesserae.dofs(smap)

        blocks = create_block_sparse_matrix(BSpline(Quadratic()), mesh; ndofs=(2, 1))
        values = Tesserae.SparseArrays.nonzeros(parent(blocks))
        values .= eachindex(values)
        extracted = @inferred extract(blocks, blockmap)
        @test Tesserae.SparseArrays.issparse(extracted)
        @test extracted == parent(blocks)[expected_dofs, expected_dofs]
        @test extract(view, blocks, blockmap) == view(parent(blocks), expected_dofs, expected_dofs)

        up = blocks[1,2]
        u_dofs = Tesserae.dofs(blockmap[1])
        p_dofs = Tesserae.dofs(blockmap[2])
        rows, cols = parentindices(up)
        extracted_up = @inferred extract(up, blockmap[1], blockmap[2])
        @test Tesserae.SparseArrays.issparse(extracted_up)
        @test extracted_up == parent(blocks)[rows[u_dofs], cols[p_dofs]]
        @test extract(view, up, blockmap[1], blockmap[2]) == view(parent(blocks), rows[u_dofs], cols[p_dofs])

        wrong_blocks = dofmap((vmask,))
        wrong_sizes = dofmap((smask, vmask))
        @test_throws DimensionMismatch extract(blocks, wrong_blocks)
        @test_throws DimensionMismatch extract(blocks, wrong_sizes)
        @test_throws ArgumentError extract(blocks, vmap)
        @test_throws ArgumentError BlockDofMap(())
        @test_throws ArgumentError dofmap(())
    end
end

@testset "FEM sparse matrix pattern" begin
    cmesh = CartesianMesh(1, (0,2), (0,1))
    geometry = FEMesh(Tesserae.Quad9(), cmesh)
    quad4 = only(generate_field_meshes((geometry,), Order(1)))
    quad9 = only(generate_field_meshes((geometry,)))

    @test_throws UndefKeywordError create_sparse_matrix(quad4)

    A = create_sparse_matrix((quad9, quad4); ndofs=(2, 1))
    B = create_sparse_matrix((quad4, quad9); ndofs=(1, 2))
    @test size(A) == (30, 6)
    @test Tesserae.SparseArrays.nnz(A) == 132

    GridPropU = @NamedTuple{x::Vec{2,Float64}, u::Vec{2,Float64}}
    GridPropP = @NamedTuple{x::Vec{2,Float64}, p::Float64}
    PointProp = @NamedTuple{x::Vec{2,Float64}, V::Float64}
    velocity_grid = generate_grid(GridPropU, quad9)
    pressure_grid = generate_grid(GridPropP, quad4)
    rule = generate_quadrature_rule(basis(geometry))
    points = generate_particles(PointProp, geometry, rule)
    velocity_weights = generate_basis_weights(quad9, size(points); name=Val(:N))
    pressure_weights = generate_basis_weights(quad4, size(points); name=Val(:N))
    update!(velocity_weights, points, quad9; geometry, measure=points.V)
    update!(pressure_weights, points, quad4; geometry)

    @P2G_Matrix (velocity_grid,pressure_grid)=>(i,j) points=>p (velocity_weights,pressure_weights)=>(ip,jp) begin
        A[i,j] = @∑ ∇N[ip] * N[jp] * V[p]
        B[j,i] = @∑ ∇N[ip] * N[jp] * V[p]
    end
    @test any(!iszero, Tesserae.SparseArrays.nonzeros(A))
    @test B ≈ A'

    parent_matrix = create_sparse_matrix((quad9, quad4); ndofs=(3, 2))
    parent_row_dofs = LinearIndices((3, length(quad9)))
    parent_col_dofs = LinearIndices((2, length(quad4)))
    row_indices = vec(parent_row_dofs[[3,1], :])
    col_indices = vec(parent_col_dofs[2:2, :])
    matrix_view = view(parent_matrix, row_indices, col_indices)
    assembler = @inferred Tesserae.matrix_assembler(matrix_view, quad9, quad4, basis(velocity_weights), basis(pressure_weights))
    @test assembler isa Tesserae.GenericMatrixAssembler

    @P2G_Matrix (velocity_grid,pressure_grid)=>(i,j) points=>p (velocity_weights,pressure_weights)=>(ip,jp) begin
        matrix_view[i,j] = @∑ ∇N[ip] * N[jp] * V[p]
    end
    @test matrix_view ≈ A

    block_matrix = @inferred create_block_sparse_matrix((quad9, quad4); ndofs=(2, 1))
    @test block_matrix[1,1] == create_sparse_matrix(quad9; ndofs=2)
    @test block_matrix[1,2] == create_sparse_matrix((quad9, quad4); ndofs=(2, 1))
    @test block_matrix[2,1] == create_sparse_matrix((quad4, quad9); ndofs=(1, 2))
    @test block_matrix[2,2] == create_sparse_matrix(quad4; ndofs=1)
    @test_throws TypeError create_block_sparse_matrix((quad9, quad4); ndofs=(2, 1, 1))
    @P2G_Matrix (velocity_grid,pressure_grid)=>(i,j) points=>p (velocity_weights,pressure_weights)=>(ip,jp) begin
        block_matrix[1,2][i,j] = @∑ ∇N[ip] * N[jp] * V[p]
        block_matrix[2,1][j,i] = @∑ ∇N[ip] * N[jp] * V[p]
    end
    @test block_matrix[1,2] ≈ A
    @test block_matrix[2,1] ≈ B

    shifted = FEMesh(Tesserae.Quad4(), CartesianMesh(1, (2,4), (0,1)))
    @test_throws ArgumentError create_sparse_matrix((quad9, shifted); ndofs=(2, 1))

    reversed_supports = supportnodes.(Ref(quad4), reverse(collect(cells(quad4))))
    reversed = FEMesh(Tesserae.Quad4(), collect(quad4), reversed_supports)
    @test_throws ArgumentError create_sparse_matrix((quad9, reversed); ndofs=(2, 1))

    partial = FEMesh(Tesserae.Quad4(), collect(quad4[supportnodes(quad4, 1)]), [Tesserae.SVector(1, 2, 3, 4)])
    @test_throws DimensionMismatch create_sparse_matrix((quad9, partial); ndofs=(2, 1))
end
