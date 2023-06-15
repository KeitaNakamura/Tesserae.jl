using IterativeSolvers
using LinearMaps: LinearMap

function default_linsolve(x, A, b; kwargs...)
    T = eltype(b)
    gmres!(fillzero!(x), A, b; maxiter=15, initially_zero=true, abstol=T(0.1), reltol=T(0.1), kwargs...)
end

struct NewtonSolver{T, F}
    maxiter::Int
    tol::T
    θ::T
    linsolve::F
    R::Vector{T}
    δv::Vector{T}
end
function NewtonSolver{T}(; maxiter::Int=50, tol::Real=sqrt(eps(T)), implicit_parameter::Real=1, linsolve=default_linsolve) where {T}
    NewtonSolver(maxiter, T(tol), T(implicit_parameter), linsolve, T[], T[])
end
NewtonSolver(; kwargs...) = NewtonSolver{Float64}(; kwargs...)

function Base.resize!(solver::NewtonSolver, n::Integer)
    resize!(solver.R, n)
    resize!(solver.δv, n)
    solver
end

isless_eps(x::Real, p::Int) = abs(x) < eps(typeof(x))^(1/p)
isconverged(x::Real, solver::NewtonSolver) = abs(x) < solver.tol

function recompute_grid_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace, alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    fillzero!(grid.f)
    parallel_each_particle(space; parallel) do p
        @inbounds begin
            pt = LazyRow(particles, p)
            itp = get_interpolation(space)
            mp = values(space, p)
            grid_to_particle!(alg, system, Val((:∇v,)), pt, grid, itp, mp)
            update_stress!(pt)
            particle_to_grid!(alg, system, Val((:f,)), grid, pt, itp, mp)
        end
    end
end

# for matrix-free linear solver
function jacobian_matrix(solver::NewtonSolver, grid::Grid, particles::Particles, space::MPSpace, Δt::Real, freedofs::Vector{<: CartesianIndex}, alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    @inline function update_stress!(pt)
        @inbounds begin
            ∇δvₚ = pt.∇v
            pt.σ = (pt.ℂ ⊡ ∇δvₚ) / pt.V
        end
    end
    LinearMap(length(freedofs)) do Jδv, δv
        @inbounds begin
            flatarray(fillzero!(grid.δv))[freedofs] .= δv
            recompute_grid_force!(update_stress!, @rename(grid, δv=>v, δf=>f), @rename(particles, δσ=>σ), space, alg, system, parallel)
            δa = view(flatarray(grid.δf ./= grid.m), freedofs)
            @. Jδv = δv - solver.θ * Δt * δa
        end
    end
end

function diagonal_preconditioner!(P::AbstractVector, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, freedofs::Vector{<: CartesianIndex}, parallel::Bool) where {dim}
    fill!(P, 1)
    fillzero!(grid.δv)
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
    @. grid.δv *= Δt / grid.m
    Diagonal(broadcast!(+, P, P, view(flatarray(grid.δv), freedofs)))
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
    θ = solver.θ

    # recompute particle stress and grid force
    @. grid.δv = (1-θ)*grid.vⁿ + θ*grid.v
    recompute_grid_force!(update_stress!, @rename(grid, δv=>v), particles, space, alg, system, parallel)

    if solver.maxiter != 0
        freedofs = filter(CartesianIndices(isfixed)) do I
            I′ = CartesianIndex(Base.tail(Tuple(I)))
            @inbounds isnonzero(grid, I′) && !iszero(grid.m[I′]) && !isfixed[I]
        end

        resize!(solver, length(freedofs))
        A = jacobian_matrix(solver, grid, particles, space, Δt, freedofs, alg, system, parallel)

        vⁿ = @inbounds view(flatarray(grid.vⁿ), freedofs)
        if !isless_eps(maximum(abs, vⁿ), 1)
            r⁰ = norm(vⁿ)
            @inbounds for k in 1:solver.maxiter
                # compute residual for Newton's method
                R = grid.δv # reuse grid.δv
                @. R = grid.v - grid.vⁿ - Δt * (grid.f/grid.m)
                solver.R .= view(flatarray(R), freedofs)
                isconverged(norm(solver.R)/r⁰, solver) && return

                # solve linear equation A⋅δv = -R
                # P = diagonal_preconditioner!(solver.P, particles, grid, space, Δt, freedofs, parallel)
                # solver.linsolve(solver.δv, A, rmul!(solver.R, -1); Pl=P)
                solver.linsolve(solver.δv, A, rmul!(solver.R, -1))
                isconverged(norm(solver.δv), solver) && return

                # update grid velocity
                v = view(flatarray(grid.v), freedofs)
                @. v += solver.δv

                # recompute particle stress and grid force
                @. grid.δv = (1-θ)*grid.vⁿ + θ*grid.v
                recompute_grid_force!(update_stress!, @rename(grid, δv=>v), particles, space, alg, system, parallel)
            end
            @warn "Newton's method not converged"
        end
    end
end
