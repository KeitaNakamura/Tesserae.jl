struct PenaltyMethod{F <: Function, dim, Grid_g <: AbstractArray{<: Vec{dim}, dim}, Grid_μ <: AbstractArray{<: Real, dim}, Grid_v <: AbstractArray{<: Vec{dim}, dim}, S, T <: Real}
    penalty_force::F
    grid_g::Grid_g
    grid_μ::Grid_μ
    grid_v::Grid_v
    storage::S
    microslip::T
end

function PenaltyMethod(
        penalty_force::Function,
        grid_g::AbstractArray{<: Vec},
        grid_μ::AbstractArray{<: Real},
        grid_v::AbstractArray{<: Vec} = FillArray(zero(eltype(grid_g)), size(grid_g));
        storage = nothing,
        microslip = 0,
    )
    PenaltyMethod(penalty_force, grid_g, grid_μ, grid_v, storage, microslip)
end

function compute_penalty_force!(grid::Grid, p::PenaltyMethod, Δt::Real)
    @assert size(grid) == size(p.grid_g) == size(p.grid_μ) == size(p.grid_v)
    fillzero!(grid.fᵖ)
    fillzero!(grid.dfᵖdu)
    for I in CartesianIndices(grid)
        if isactive(grid, I)
            i = nonzeroindex(grid, I)
            g⁰ = p.grid_g[i]
            if !iszero(g⁰)
                n = normalize(g⁰)
                μ = p.grid_μ[i]
                v_rigid = p.grid_v[i]
                dfᵖdu, fᵖ = gradient(grid.u[i], :all) do u
                    # normal
                    g = g⁰ + normal(u-v_rigid*Δt, n)
                    fₙ = -p.penalty_force(g⋅n) * n

                    # tangential
                    uₜ = tangential(u-v_rigid*Δt, n)
                    (iszero(μ) || iszero(uₜ) || iszero(fₙ)) && return fₙ

                    if iszero(p.microslip)
                        fₜ★ = grid.m[i] * uₜ / Δt^2
                        fₜ = -min(1, μ*norm(fₙ)/norm(fₜ★)) * fₜ★
                    else
                        ξ = norm(uₜ) / p.microslip
                        α = ξ<1 ? -ξ^2+2ξ : one(ξ)
                        fₜ = -α*μ*norm(fₙ)*normalize(uₜ)
                    end

                    fₙ + fₜ
                end
                grid.fᵖ[i] = fᵖ
                grid.dfᵖdu[i] = dfᵖdu
            end
        end
    end
end
