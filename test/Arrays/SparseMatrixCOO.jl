@testset "SparseMatrixCOO" begin
    for T in (Float32, Float64)
        (@inferred SparseMatrixCOO())::SparseMatrixCOO{Float64}
        A = (@inferred SparseMatrixCOO{T}())::SparseMatrixCOO{T}
        mat = [1 2; 3 4]
        I = [2, 3]
        J = [3, 4]
        @testset "push! matrix" begin
            push!(A, mat, I, J)
            ## without size
            sparse!(A)
            res = zeros(3, 4)
            res[I, J] .= mat
            @test (@inferred sparse(A))::SparseMatrixCSC{T, Int} == res
            ## with size
            sparse!(A, 4, 4)
            res = zeros(4, 4)
            res[I, J] .= mat
            @test (@inferred sparse(A))::SparseMatrixCSC{T, Int} == res
        end
        @testset "push! vector (diagonal)" begin
            vec = [1, 2]
            K = [1, 3]
            push!(A, vec, K)
            sparse!(A)
            res = zeros(3, 4)
            res[I, J] .= mat
            res[K, K] .+= Diagonal(vec)
            @test (@inferred sparse(A))::SparseMatrixCSC{T, Int} == res
        end
        empty!(A)
        @test isempty(A.I)
        @test isempty(A.J)
        @test isempty(A.V)
    end
end
