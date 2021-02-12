module AssemblingTest

using Jams
using Test, LinearAlgebra

function check_point_to_grid(x, N)
    @test (@inferred x[1]) |> Array |> ndims == N
end

function check_grid_to_point(x, T)
    # TODO: fix type instability
    # (@inferred x[1])::T
end

@testset "Point-to-grid assembling" begin
    grid = Grid(0:0.1:1.0, 0:0.1:1.0)
    space = MPSpace(LinearBSpline(dim=2), grid, 10)
    xₚ = pointstate(space, Vec{2, Float64})
    for i in 1:length(xₚ)
        xₚ[i] = rand(Vec{2})
    end
    N = construct(:shape_value, space)
    reinit!(space, xₚ)

    mₚ = pointstate(space, Float64)
    check_point_to_grid((@inferred ∑ₚ(N)), 1)
    check_point_to_grid((@inferred ∑ₚ(N*xₚ)), 1)
    check_point_to_grid((@inferred ∑ₚ(N*N)), 2)
    check_point_to_grid((@inferred ∑ₚ(N*mₚ*N)), 2)
    check_point_to_grid((@inferred ∑ₚ(∇(N)⋅∇(N))), 2)
    check_point_to_grid((@inferred ∑ₚ(∇(N)*mₚ⋅∇(N))), 2)

    vₚ = pointstate(space, Vec{2, Float64})
    check_grid_to_point((@inferred ∑ᵢ(N*vₚ)), Vec{2,Float64})
end

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
    A = gridstate_matrix(space, Mat{2,2,Float64}); reinit!(A)
    ex = ∑ₚ(∇(N) ⊗ ∇(N))
    (@inferred ex[1])::AbstractCollection{-1}
    A ← ex
    B = SparseMatrixCOO()
    for p in 1:npoints(space)
        ∇Nᵢ = vcat(∇(N)[p]...)
        inds = MPSpaces.dofindices(space, p; dof)
        push!(B, ∇Nᵢ*∇Nᵢ', inds, inds)
    end
    @test sparse(A) == sparse!(B)

    # Test 2
    cₚ = pointstate(space, SymmetricFourthOrderTensor{2})
    for i in 1:length(cₚ)
        cₚ[i] = rand(SymmetricFourthOrderTensor{2})
    end
    A = gridstate_matrix(space, Mat{2,2,Float64}); reinit!(A)
    ex = ∑ₚ(dotdot(∇(N), cₚ, ∇(N)))
    (@inferred ex[1])::AbstractCollection{-1}
    A ← ex
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
    @test sparse(A) ≈ sparse!(B)

    # Test 3 (mass matrix)
    A = gridstate_matrix(space, Mat{2,2,Float64}); reinit!(A)
    B = gridstate_matrix(space, Mat{2,2,Float64}); reinit!(B)
    mᵢ = gridstate(space, Float64)
    # consisten mass matrixt
    ex = ∑ₚ(N*N)
    (@inferred ex[1])::AbstractCollection{-1}
    A ← ex
    # lumped mass matrix (using gridstate)
    ex = ∑ₚ(N)
    (@inferred ex[1])::AbstractCollection{0}
    mᵢ ← ex
    B ← GridDiagonal(mᵢ)
    @test Diagonal(vec(sum(sparse(A), dims = 2))) ≈ sparse(B)
    # lumped mass matrix (without temporary array)
    B ← GridDiagonal(∑ₚ(N))
    @test Diagonal(vec(sum(sparse(A), dims = 2))) ≈ sparse(B)
end

end
