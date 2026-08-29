@testset "Stencil" begin
    S = Tesserae.Stencil

    # A first-order stencil is exact on a linear field, so every entry of the
    # gradient must reproduce the field's constant coefficient matrix.
    @testset "Gradient Face-to-Cell" begin
        A = [0.3 0.1; -0.2 0.5]
        h = 0.1
        n = 8
        field(I) = Vec(A[1,1]*I[1] + A[1,2]*I[2], A[2,1]*I[1] + A[2,2]*I[2]) * h
        srcs = (S.StencilArray(S.Face(1), [field(I) for I in CartesianIndices((n+1, n))]),
                S.StencilArray(S.Face(2), [field(I) for I in CartesianIndices((n, n+1))]))
        expected = Tensor{Tuple{2,2,2}}((a, i, j) -> A[a,j])
        dest = S.StencilArray(S.Cell(), fill(zero(expected), n, n))
        S.stencil!(S.Gradient{1}(), dest, srcs; pad=1, spacing=h)
        @test all(t -> t ≈ expected, S.inner(dest; pad=1))
    end
end
