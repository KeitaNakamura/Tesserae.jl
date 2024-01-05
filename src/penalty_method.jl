struct PenaltyMethod{F <: Function, dim, Grid_g <: AbstractArray{<: Vec{dim}, dim}, Grid_μ <: AbstractArray{<: Real, dim}, Grid_v <: AbstractArray{<: Vec{dim}, dim}, S, T <: Real}
    penalty_force::F
    grid_g::Grid_g
    grid_μ::Grid_μ
    grid_v::Grid_v
    storage::S
    microslip::T
    microslip_auto_parameter::T
end

function PenaltyMethod(
        penalty_force::Function,
        grid_g::AbstractArray{<: Vec},
        grid_μ::AbstractArray{<: Real},
        grid_v::AbstractArray{<: Vec} = FillArray(zero(eltype(grid_g)), size(grid_g));
        storage = nothing,
        microslip = NaN,
        microslip_auto_parameter = 0.1,
    )
    PenaltyMethod(penalty_force, grid_g, grid_μ, grid_v, storage, microslip, microslip_auto_parameter)
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
                    # relative displacement
                    uᵣ = u - Δt * v_rigid

                    # normal
                    g = g⁰ + normal(uᵣ, n)
                    c = p.penalty_force(g ⋅ n)
                    fₙ = -c * n

                    # tangential
                    uₜ = tangential(uᵣ, n)
                    (iszero(fₙ) || iszero(μ) || iszero(uₜ)) && return fₙ

                    if isnan(p.microslip) # default
                        ϵᵤ = norm(p.microslip_auto_parameter*g + uₜ)
                    else
                        ϵᵤ = p.microslip
                    end
                    if isone(-ϵᵤ) # special treatment to force simple stick condition
                        fₜ = -grid.m[i] * uₜ / Δt^2
                    else
                        uₜ_norm = norm(uₜ)
                        ξ = uₜ_norm / ϵᵤ
                        α = ξ<1 ? -ξ^2+2ξ : one(ξ)
                        fₜ = -α*μ*c*(uₜ/uₜ_norm)
                    end

                    fₙ + fₜ
                end
                grid.fᵖ[i] = fᵖ
                grid.dfᵖdu[i] = dfᵖdu
            end
        end
    end
end
