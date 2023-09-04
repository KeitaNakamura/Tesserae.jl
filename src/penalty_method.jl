struct PenaltyMethod{F <: Function, dim, Grid_g <: AbstractArray{<: Vec{dim}, dim}, Grid_μ <: AbstractArray{<: Real, dim}, Grid_v <: AbstractArray{<: Vec{dim}, dim}, S}
    penalty_force::F
    grid_g::Grid_g
    grid_μ::Grid_μ
    grid_v::Grid_v
    storage::S
end

function PenaltyMethod(
        penalty_force::Function,
        grid_g::AbstractArray{<: Vec},
        grid_μ::AbstractArray{<: Real},
        grid_v::AbstractArray{<: Vec} = FillArray(zero(eltype(grid_g)), size(grid_g));
        storage = nothing,
    )
    PenaltyMethod(penalty_force, grid_g, grid_μ, grid_v, storage)
end

function compute_penalty_force!(grid::Grid, Δt::Real, p::PenaltyMethod)
    compute_penalty_force!(grid, Δt, p.penalty_force, p.grid_g, p.grid_μ, p.grid_v)
end

function compute_penalty_force!(grid::Grid, Δt::Real, penalty_force::Function, grid_g::AbstractArray{<: Vec}, grid_μ::AbstractArray{<: Real}, grid_v::AbstractArray{<: Vec})
    @assert size(grid) == size(grid_g) == size(grid_μ) == size(grid_v)
    fillzero!(grid.fᵖ)
    fillzero!(grid.dfᵖdv)
    fillzero!(grid.dfᵖdf)
    for I in CartesianIndices(grid)
        if isactive(grid, I)
            i = nonzeroindex(grid, I)
            g⁰ = grid_g[i]
            if !iszero(norm(g⁰))
                n = normalize(g⁰)
                μ = grid_μ[i]
                v_rigid = grid_v[i]
                f̄ₜ = tangential(grid.m[i]*(grid.vⁿ[i]-v_rigid)/Δt + grid.fext[i], n)
                (dfᵖdv,dfᵖdf), fᵖ = gradient(grid.v[i], grid.fint[i], :all) do v, f
                    # normal
                    g = g⁰ + normal(v-v_rigid, n) * Δt
                    fₙ = penalty_force(g)

                    # tangential
                    iszero(μ) && return fₙ
                    fₜ★ = tangential(f,n) + f̄ₜ
                    fₜ = -min(1, μ*norm(fₙ)/norm(fₜ★)) * fₜ★

                    fₙ + fₜ
                end
                grid.fᵖ[i] += fᵖ
                grid.dfᵖdv[i] += dfᵖdv
                grid.dfᵖdf[i] += dfᵖdf
            end
        end
    end
end
