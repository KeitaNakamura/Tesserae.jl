using IterativeSolvers
using LinearMaps: LinearMap

function default_linsolve(x, A, b; kwargs...)
    gmres!(fillzero!(x), A, b; maxiter=15, initially_zero=true, abstol=sqrt(eps(eltype(b))), kwargs...)
end

struct NewtonSolver{T, F}
    maxiter::Int
    tol::T
    linsolve::F
    R::Vector{T}
    δv::Vector{T}
end
function NewtonSolver{T}(; maxiter::Int=50, tol::Real=sqrt(eps(T)), linsolve=default_linsolve) where {T}
    NewtonSolver(maxiter, tol, linsolve, T[], T[])
end
NewtonSolver(; kwargs...) = NewtonSolver{Float64}(; kwargs...)

function Base.resize!(solver::NewtonSolver, n::Integer)
    resize!(solver.R, n)
    resize!(solver.δv, n)
    solver
end

isless_eps(x::Real, p::Int) = abs(x) < eps(typeof(x))^(1/p)
isconverged(x::Real, solver::NewtonSolver) = abs(x) < solver.tol

# for matrix-free linear solver
function jacobian_matrix(particles::Particles, grid::Grid, space::MPSpace, Δt::Real, freedofs::Vector{<: CartesianIndex}; alg::TransferAlgorithm=FLIP(), system::CoordinateSystem=DefaultSystem(), parallel::Bool=true)
    LinearMap(length(freedofs)) do Jδv, δv
        @inbounds begin
            # grid-to-particle to compute δvₚ from δvᵢ, then compute Cauchy stress increment
            flatarray(fillzero!(grid.δv))[freedofs] .= δv
            grid_to_particle!(:∇v, particles, @rename(grid, δv=>v, v=>_), space; alg, system, parallel) do pt
                @_inline_meta
                @inbounds begin
                    δvₚ = pt.∇v
                    pt.δσ = (pt.ℂ ⊡ δvₚ) / pt.V
                end
            end

            # particle-to-grid to compute δfᵢ (i.e., ∂fᵢ∂δvᵢ ⋅ δvᵢ)
            δf = fillzero!(grid.δv) # reuse grid.δv
            particle_to_grid!(:f, @rename(grid, δv=>f, f=>_), @rename(particles, δσ=>σ, σ=>_), space; alg, system, parallel)

            # compute J⋅δvᵢ (matrix-vector product)
            δa = view(flatarray(δf ./= grid.m), freedofs)
            @. Jδv = δv - Δt * δa
        end
    end
end

function diagonal_preconditioner!(P::AbstractVector, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, freedofs::Vector{<: CartesianIndex}; parallel::Bool=true) where {dim}
    fill!(P, 1)
    parallel_each_particle(space; parallel) do p
        @_inline_meta
        @inbounds begin
            ℂₚ = Tensorial.resizedim(particles.ℂ[p], Val(dim))
            mp = values(space, p)
            gridindices = neighbornodes(mp, grid)
            @simd for j in CartesianIndices(gridindices)
                i = gridindices[j]
                ∇N = mp.∇N[j]
                grid.δv[i] += diag(∇N ⋅ ℂₚ ⋅ ∇N) # reuse δv
            end
        end
    end
    @. grid.δv *= -Δt / grid.m
    Diagonal(broadcast!(-, P, P, view(flatarray(grid.δv), freedofs)))
end

# implicit version of grid_to_particle!
function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::NewtonSolver, isfixed::AbstractArray{Bool}; parallel::Bool) where {names}
    @assert :∇v in names
    grid_to_particle!(update_stress!, :∇v, particles, grid, space, Δt, solver, isfixed; alg, system, parallel)
    rest = tuple(delete!(Set(names), :∇v)...)
    grid_to_particle!(rest, particles, grid, space, Δt; alg, system, parallel)
end

function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{(:∇v,)}, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, solver::NewtonSolver{T}, isfixed::AbstractArray{Bool}; parallel::Bool) where {dim, T}
    @assert size(isfixed) == (dim, size(grid)...)

    grid_to_particle!(update_stress!, :∇v, particles, grid, space; alg, system, parallel)

    if solver.maxiter != 0
        freedofs = filter(CartesianIndices(isfixed)) do I
            I′ = CartesianIndex(Base.tail(Tuple(I)))
            @inbounds isnonzero(grid, I′) && !iszero(grid.m[I′]) && !isfixed[I]
        end

        resize!(solver, length(freedofs))
        A = jacobian_matrix(particles, grid, space, Δt, freedofs; alg, system, parallel)

        vⁿ = @inbounds view(flatarray(grid.vⁿ), freedofs)
        if !isless_eps(maximum(abs, vⁿ), 1)
            ok = false
            r⁰ = norm(vⁿ)
            @inbounds for k in 1:solver.maxiter
                # compute grid force at k iterations
                fillzero!(grid.f)
                particle_to_grid!(:f, grid, particles, space; alg, system, parallel)

                # compute residual for Newton's method
                R = grid.δv # reuse grid.δv
                @. R = grid.v - grid.vⁿ - Δt * (grid.f/grid.m)
                solver.R .= view(flatarray(R), freedofs)
                isconverged(norm(solver.R)/r⁰, solver) && (ok=true; break)

                # solve linear equation A⋅δv = -R
                # P = diagonal_preconditioner!(solver.P, particles, grid, space, Δt, freedofs; parallel)
                # solver.linsolve(solver.δv, A, rmul!(solver.R, -1); Pl=P)
                solver.linsolve(solver.δv, A, rmul!(solver.R, -1))
                isconverged(norm(solver.δv), solver) && (ok=true; break)

                # update grid velocity
                v = view(flatarray(grid.v), freedofs)
                @. v += solver.δv

                # recompute particle stress from grid velocity
                grid_to_particle!(update_stress!, :∇v, particles, grid, space; alg, system, parallel)
            end
            ok || @warn "Newton's method not converged"
        end
    end
end
