using Tesserae
using Metal
using SparseArrays
using Test

Metal.allowscalar(false)

@testset "Metal sparse transfers" begin
    for device in (Tesserae.MetalDevice{Tesserae.CastFloat32}(),
                   Tesserae.MetalDevice{Tesserae.PreserveEltype}())
        for A in (spzeros(Float32, 3, 3), spzeros(Float64, 3, 3),
                  spzeros(Vec{2,Float64}, 3, 3), sparsevec([1], [1.0], 3))
            @test_throws ArgumentError device(A)
        end

        blocks = create_block_sparse_matrix(Float32, BSpline(Linear()), CartesianMesh(1.0, (0,2)); ndofs=(1, 1))
        @test_throws ArgumentError device(blocks)
        @test_throws ArgumentError device(blocks[1,1])
    end
end
