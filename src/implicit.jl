using IterativeSolvers
using LinearMaps: LinearMap

struct NewtonMethod{T, F}
    maxiter::Int
    linsolve::F
    R::Vector{T}
    δv::Vector{T}
end
function NewtonMethod{T}(;maxiter::Int = 20,
                          linsolve = (x, A, b) -> gmres!(fillzero!(x), A, b; maxiter=20, initially_zero=true)) where {T}
    NewtonMethod(maxiter, linsolve, T[], T[])
end
NewtonMethod(; kwargs...) = NewtonMethod{Float64}(; kwargs...)

# for matrix-free linear solver
function get_Jδv(particles::Particles, grid::Grid, space::MPSpace, Δt::Real, freedofs::Vector{<: CartesianIndex}; alg::TransferAlgorithm=FLIP(), system::CoordinateSystem=DefaultSystem(), parallel::Bool=true)
    LinearMap(length(freedofs)) do Jδv, δv
        @inbounds begin
            # grid-to-particle to compute δvₚ from δvᵢ, then compute Cauchy stress increment
            flatarray(fillzero!(grid.δv))[freedofs] .= δv
            grid_to_particle!(:∇v, particles, @rename(grid, δv=>v, v=>_), space; alg, system, parallel) do pt
                @_inline_meta
                δvₚ = pt.∇v
                pt.δσ = pt.ℂ ⊡ δvₚ
            end

            # particle-to-grid to compute δfᵢ (i.e., ∂fᵢ∂δvᵢ ⋅ δvᵢ)
            δf = grid.δv # reuse grid.δv
            fillzero!(δf)
            particle_to_grid!(:f, @rename(grid, δv=>f, f=>_), @rename(particles, δσ=>σ, σ=>_), space; alg, system, parallel)

            # compute J⋅δvᵢ (matrix-vector product)
            δa = view(flatarray(δf ./= grid.m), freedofs)
            @. Jδv = δv - Δt * δa
        end
    end
end

# implicit version of grid_to_particle!
function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::NewtonMethod, fixeddofs::AbstractArray{Bool}; parallel::Bool) where {names}
    @assert :∇v in names
    grid_to_particle!(update_stress!, :∇v, particles, grid, space, Δt, solver, fixeddofs; alg, system, parallel)
    rest = tuple(delete!(Set(names), :∇v)...)
    grid_to_particle!(rest, particles, grid, space, Δt; alg, system, parallel)
end

function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{(:∇v,)}, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, solver::NewtonMethod{T}, fixeddofs::AbstractArray{Bool}; parallel::Bool) where {dim, T}
    @assert size(fixeddofs) == (dim, size(grid)...)

    grid_to_particle!(update_stress!, :∇v, particles, grid, space; alg, system, parallel)

    if solver.maxiter != 0
        isconverged(x::Real, p::Int) = x < eps(typeof(x))^(1/p)

        freedofs = filter(CartesianIndices(fixeddofs)) do I
            I′ = CartesianIndex(Base.tail(Tuple(I)))
            @inbounds isnonzero(grid, I′) && !iszero(grid.m[I′]) && !fixeddofs[I]
        end

        resize!(solver.R, length(freedofs))
        resize!(solver.δv, length(freedofs))
        Jδv = get_Jδv(particles, grid, space, Δt, freedofs; alg, system, parallel)

        vⁿ = view(flatarray(grid.vⁿ), freedofs)
        if !isconverged(maximum(abs, vⁿ), 1)
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
                isconverged(norm(solver.R) / r⁰, 2) && (ok=true; break)

                # solve linear equation
                solver.linsolve(solver.δv, Jδv, rmul!(solver.R, -1))

                # update grid velocity
                v = view(flatarray(grid.v), freedofs)
                @. v += solver.δv

                # recompute particle stress from grid velocity
                grid_to_particle!(update_stress!, :∇v, particles, grid, space; alg, system, parallel)
            end
            ok || @warn "not converged in Newton's method"
        end
    end

    particles.Fⁿ .= particles.F
end
