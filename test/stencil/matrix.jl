using SparseArrays

@testset "Fixed stencil patterns and paired assembly" begin
    for counts in ((1,), (5,), (4, 3), (4, 3, 2))
        C = Region(Cell(), counts)
        directions = unitoffsets(C)
        offsets = (0*first(directions), directions..., map(-, directions)...)
        A = create_sparse_matrix(C, C, offsets; ndofs=1)
        structure = (copy(A.colptr), copy(rowvals(A)))
        expected = zeros(prod(counts), prod(counts))
        ids = LinearIndices(counts)
        pressure = reshape(sin.(1.0:prod(counts)), counts)
        divergence = zeros(counts)
        for (d, e) in enumerate(directions)
            F = Region(Face(d), C)
            I = interior(F)
            L, R = I-e/2, I+e/2
            coefficient = reshape(collect(1.0:prod(size(F))), size(F))
            @stencil_assemble begin
                A[L, L] += coefficient[I]
                A[L, R] -= coefficient[I]
                A[R, L] -= coefficient[I]
                A[R, R] += coefficient[I]
            end
            step = CartesianIndex(ntuple(j -> Int(j == d), length(counts)))
            for K in CartesianIndices(counts)
                K[d] == counts[d] && continue
                i, j = ids[K], ids[K+step]
                w = coefficient[K+step]
                expected[i, i] += w
                expected[i, j] -= w
                expected[j, i] -= w
                expected[j, j] += w
            end
            gradient = zeros(size(F))
            @stencil begin
                gradient[I] = coefficient[I]*(pressure[I+e/2]-pressure[I-e/2])
                divergence[C] -= gradient[C+e/2]-gradient[C-e/2]
            end
        end
        @test Matrix(A) == expected
        @test A*vec(pressure) ≈ vec(divergence) rtol=8eps()
        @test issymmetric(A)
        @test A*ones(prod(counts)) == zeros(prod(counts))
        @test (A.colptr, rowvals(A)) == structure
        fillzero!(A)
        @test all(iszero, nonzeros(A))
        @test (A.colptr, rowvals(A)) == structure
    end
end

@testset "Local matrices are evaluated once per selected point" begin
    cells = Region(Cell(), (4,))
    (e,) = unitoffsets(cells)
    matrix = create_sparse_matrix(cells, cells, (e,); ndofs=1)
    r = 1:4
    @stencil_assemble matrix[cells, cells+e] = r[cells]+1
    @test diag(matrix, 1) == [2, 3, 4]
    x = [1.0, 4.0, 9.0, -1.0]
    @stencil_assemble matrix[cells, cells+e] = sqrt(x)
    @test diag(matrix, 1) == [1, 2, 3]
    @stencil_assemble matrix[cells, cells+4e] = error("empty selection evaluated")

    C = Region(Cell(), (4, 3))
    ex, ey = unitoffsets(C)
    source = reshape(collect(1:prod(size(C))), size(C))
    for (nr, nc) in ((2, 1), (1, 2), (2, 2), (3, 3))
        A = create_sparse_matrix(C, C, (0ex,); ndofs=(nr, nc))
        calls = zeros(Int, length(source))
        block(x) = (calls[x] += 1; [x+a+10b for a in 1:nr, b in 1:nc])
        @stencil_assemble A[C, C] = block(source[C+ex])
        @test reshape(calls, size(C)) == vcat(zeros(Int, 1, 3), ones(Int, 3, 3))
        ids = LinearIndices(size(C))
        for I in CartesianIndices(size(C))
            expected = I[1] == size(C, 1) ? zeros(nr, nc) :
                [source[I+CartesianIndex(1, 0)]+a+10b for a in 1:nr, b in 1:nc]
            i = ids[I]
            @test A[nr*(i-1)+1:nr*i, nc*(i-1)+1:nc*i] == expected
        end
    end
end

@testset "Local matrix assembly snapshots aliased inputs" begin
    C = Region(Cell(), (4,))
    reversed = reflect(C, AxisPlane(C.frame, 1, 2//1))
    block(x) = Mat{2, 2}((x, 2x, 3x, 4x))
    for op in (identity, +, -)
        A = reshape(collect(1.0:64.0), 8, 8)
        source = view(vec(A), 1:4)
        expected = copy(A)
        initial = reverse(copy(source))
        for i in 1:4
            rows = 2i-1:2i
            value = block(initial[i])
            expected[rows, rows] = op === identity ? value : op(expected[rows, rows], value)
        end
        if op === identity
            @stencil_assemble A[C, C] = block(source[reversed])
        elseif op === (+)
            @stencil_assemble A[C, C] += block(source[reversed])
        else
            @stencil_assemble A[C, C] -= block(source[reversed])
        end
        @test A == expected
    end
end

@testset "Staggered placements, local blocks and sparse views" begin
    C = Region(Cell(), (4, 3))
    ex, ey = unitoffsets(C)
    F = Region(Face(1), C)
    I = interior(F)
    L, R = I-ex/2, I+ex/2
    A = create_sparse_matrix(Float32, C, F, (-ex/2, ex/2); ndofs=(2, 1))
    v = Vec(2.0f0, -3.0f0)
    @stencil_assemble A[C, C+ex/2] = $Ref(v)
    ids = LinearIndices(size(C))
    face_ids = LinearIndices(size(F))
    for K in CartesianIndices(size(C))
        @test A[2ids[K]-1:2ids[K], face_ids[K+CartesianIndex(1, 0)]] == v
    end
    @test eltype(A) === Float32
    B = create_sparse_matrix(C, C, (0ex, -ex, ex); ndofs=2)
    block = Mat{2, 2}((1.0, 2.0, 3.0, 4.0))
    @stencil_assemble B[L, R] += $Ref(block)
    @stencil_assemble B[R, L] -= $Ref(block)
    @test B[1:2, 3:4] == block
    @test B[3:4, 1:2] == -block
    entries = fill(block, size(C))
    @stencil_assemble B[C, C] = entries[C]
    @test B[1:2, 1:2] == block
    @test_throws DimensionMismatch @stencil_assemble B[C, C] = 1.0
    scalar = create_sparse_matrix(C, C, (0ex, -ex, ex); ndofs=1)
    selection = view(B, 1:2:size(B, 1), 2:2:size(B, 2))
    @stencil_assemble selection[C, C] = 7
    @stencil_assemble scalar[C, C] = 7
    @test diag(selection) == diag(scalar)

    for wrap in (transpose, adjoint)
        parent_matrix = create_sparse_matrix(ComplexF64, C, C, (0ex, ex); ndofs=1)
        wrapped = wrap(parent_matrix)
        @stencil_assemble wrapped[C, C-ex] = 2+3im
        @test wrapped[2, 1] == 2+3im
        @test parent_matrix[1, 2] == wrap(2+3im)
        slots = Int[]
        Stencil.stencil_assemble!(+, wrapped, C, C-ex, 1+im, slots)
        @test length(slots) == (size(C, 1)-1)*size(C, 2)
        @test wrapped[2, 1] == 3+4im
        saved = copy(parent_matrix)
        @test_throws ArgumentError @stencil_assemble wrapped[C, C+ex] = 1
        @test parent_matrix == saved && nnz(parent_matrix) == nnz(saved)
    end
end

@testset "Common reads, order and fixed-pattern errors" begin
    C = Region(Cell(), (4, 3))
    ex, ey = unitoffsets(C)
    A = create_sparse_matrix(C, C, (0ex, ex, -ex); ndofs=1)
    source = reshape(collect(1.0:12.0), size(C))
    @stencil_assemble A[C, C] = source[C+ex]
    @test reshape(diag(A), size(C)) == vcat(source[2:end, :], zeros(1, 3))
    reversed = reflect(C, AxisPlane(C.frame, 1, 2//1))
    @stencil_assemble A[reversed, reversed] += source[C]
    @test reshape(diag(A), size(C)) == vcat(source[2:end, :], zeros(1, 3))+reverse(source; dims=1)
    before = copy(A)
    @test_throws ArgumentError @stencil_assemble A[C, C+ey] += 1
    @test A == before
    @test nnz(A) == nnz(before)
    @test_throws DimensionMismatch @stencil_assemble A[C, interior(Region(Face(1), C))] = 1
    @test_throws DimensionMismatch @stencil_assemble zeros(5, 5)[C, C] = 1
    other = Region(Cell(), CartesianMesh(1.0, (2, 6), (0, 3)))
    @test_throws ArgumentError @stencil_assemble A[C, other] = 1
    @test_throws ArgumentError @stencil_assemble A[C, C] = source[other]
    @test_throws ArgumentError create_sparse_matrix(C, C, (ex/2,); ndofs=1)
    @test_throws ArgumentError create_sparse_matrix(C, C, (0ex,); ndofs=0)
    @test_throws ArgumentError create_sparse_matrix(C, C, (0ex,); ndofs=(1, 2, 3))
    @test_throws ArgumentError create_sparse_matrix(C+ex, C, (0ex,); ndofs=1)
    @test_throws DimensionMismatch create_sparse_matrix(C, Region(Cell(), (5, 3)), (0ex,); ndofs=1)
    @test_throws ArgumentError macroexpand(@__MODULE__, :(@stencil_assemble A[C] += 1))
    @test_throws ArgumentError macroexpand(@__MODULE__, :(@stencil_assemble A[C, C] *= 2))
    @test_throws ArgumentError macroexpand(@__MODULE__, :(@stencil_assemble begin println("not an equation") end))
    empty = intersect(C, C+10ex)
    @stencil_assemble A[empty, empty] += 1
    @test A == before
    calls = Ref(0)
    target() = (calls[] += 1; A)
    @stencil_assemble target()[C, C] = 3
    @test calls[] == 1
    @test all(==(3), diag(A))
    missing = copy(A)
    missing[end, end] = 0
    dropzeros!(missing)
    before = copy(missing)
    @test_throws ArgumentError @stencil_assemble missing[C, C] += 1
    @test missing == before
    diagonal = create_sparse_matrix(C, C, (0ex, 0ex); ndofs=1)
    @test nnz(diagonal) == prod(size(C))
    @stencil_assemble diagonal[C, C] = source[C]
    aliased = reshape(nonzeros(diagonal), size(C))
    @stencil_assemble diagonal[C, C] = aliased[reversed]
    @test reshape(diag(diagonal), size(C)) == reverse(source; dims=1)
end
