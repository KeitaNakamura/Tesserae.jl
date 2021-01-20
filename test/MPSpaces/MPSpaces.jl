@testset "Assembling global matrix" begin
    grid = Grid(0:0.1:1.0, 0:0.1:1.0)
    space = MPSpace(LinearBSpline(dim=2), grid, 10)
    xₚ = pointstate(space, Vec{2, Float64})
    for i in 1:length(xₚ)
        xₚ[i] = rand(Vec{2})
    end
    N = construct(:shape_value, space)
    reinit!(space, xₚ)
    dof = 2

    # Test 1
    A = gridstate_matrix(space, Mat{2,2,Float64})
    A ← ∑ₚ(∇(N) ⊗ ∇(N))
    B = SparseMatrixCOO()
    for p in 1:npoints(space)
        ∇Nᵢ = vcat(∇(N)[p]...)
        inds = MPSpaces.dofindices(space, p; dof)
        push!(B, ∇Nᵢ*∇Nᵢ', inds, inds)
    end
    @test sparse(A) == sparse(B)

    # Test 2
    cₚ = pointstate(space, SymmetricFourthOrderTensor{2})
    for i in 1:length(cₚ)
        cₚ[i] = rand(SymmetricFourthOrderTensor{2})
    end
    A = gridstate_matrix(space, Mat{2,2,Float64})
    A ← ∑ₚ(dotdot(∇(N), cₚ, ∇(N)))
    B = SparseMatrixCOO()
    for p in 1:npoints(space)
        Nᵢ = N[p]
        c = cₚ[p]
        inds = MPSpaces.dofindices(space, p; dof)
        K = zeros(length(inds), length(inds))
        for i in eachindex(inds)
            ∇u = ∇(vec(Nᵢ)[i])
            for j in eachindex(inds)
                ∇v = ∇(vec(Nᵢ)[j])
                K[i,j] = ∇u ⊡ c ⊡ ∇v
            end
        end
        push!(B, K, inds, inds)
    end
    @test sparse(A) ≈ sparse(B)

    # Test 3 (mass matrix)
    A = gridstate_matrix(space, Mat{2,2,Float64})
    B = gridstate_matrix(space, Mat{2,2,Float64})
    mᵢ = gridstate(space, Float64)
    # consisten mass matrixt
    A ← ∑ₚ(N*N)
    # lumped mass matrix (using gridstate)
    mᵢ ← ∑ₚ(N)
    B ← GridDiagonal(mᵢ)
    @test Diagonal(vec(sum(sparse(A), dims = 2))) ≈ sparse(B)
    # lumped mass matrix (without temporary array)
    B ← GridDiagonal(∑ₚ(N))
    @test Diagonal(vec(sum(sparse(A), dims = 2))) ≈ sparse(B)
end
