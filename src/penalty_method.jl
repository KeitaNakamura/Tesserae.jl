struct PenaltyMethod{F <: Function, dim, Grid_g <: AbstractArray{<: Vec{dim}, dim}, Grid_μ <: AbstractArray{<: Real, dim}, Grid_v <: AbstractArray{<: Vec{dim}, dim}, S, T <: Union{Nothing, Real}}
    penalty_force::F
    grid_g::Grid_g
    grid_μ::Grid_μ
    grid_v::Grid_v
    storage::S
    velocity_thresh::T
end

function PenaltyMethod(
        penalty_force::Function,
        grid_g::AbstractArray{<: Vec},
        grid_μ::AbstractArray{<: Real},
        grid_v::AbstractArray{<: Vec} = FillArray(zero(eltype(grid_g)), size(grid_g));
        storage = nothing,
        velocity_thresh = nothing,
    )
    PenaltyMethod(penalty_force, grid_g, grid_μ, grid_v, storage, velocity_thresh)
end

function compute_penalty_force!(grid::Grid, integrator::EulerIntegrator, p::PenaltyMethod, Δt::Real)
    @assert size(grid) == size(p.grid_g) == size(p.grid_μ) == size(p.grid_v)
    fillzero!(grid.fᵖ)
    fillzero!(grid.dfᵖdv)
    fillzero!(grid.dfᵖdf)
    for I in CartesianIndices(grid)
        if isactive(grid, I)
            i = nonzeroindex(grid, I)
            g⁰ = p.grid_g[i]
            if !iszero(norm(g⁰))
                n = normalize(g⁰)
                μ = p.grid_μ[i]
                v_rigid = p.grid_v[i]
                f̄ₜ = tangential(grid.m[i]*(grid.vⁿ[i]-v_rigid)/Δt + grid.fext[i], n)
                (dfᵖdv,dfᵖdf), fᵖ = gradient(grid.v[i], grid.fint[i], :all) do v, f
                    # normal
                    g = g⁰ + normal(v-v_rigid, n) * Δt
                    fₙ = p.penalty_force(g)

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


function compute_penalty_force!(grid::Grid, integrator::NewmarkIntegrator, p::PenaltyMethod, Δt::Real)
    @assert size(grid) == size(p.grid_g) == size(p.grid_μ) == size(p.grid_v)
    γ, β = integrator.γ, integrator.β
    fillzero!(grid.fᵖ)
    fillzero!(grid.dfᵖdu)
    fillzero!(grid.dfᵖdf)
    for I in CartesianIndices(grid)
        if isactive(grid, I)
            i = nonzeroindex(grid, I)
            g⁰ = p.grid_g[i]
            if !iszero(norm(g⁰))
                n = normalize(g⁰)
                μ = p.grid_μ[i]
                v_rigid = p.grid_v[i]
                if p.velocity_thresh === nothing
                    f̄ₜ = tangential(grid.m[i]*(grid.vⁿ[i]-v_rigid)/Δt/β + grid.m[i]*(1/2β-1)*grid.aⁿ[i] + grid.fext[i], n)
                else
                    ūₜ = tangential((grid.vⁿ[i]-v_rigid)*Δt + ((1/2-β)*grid.aⁿ[i] + β*grid.fext[i]/grid.m[i])*Δt^2, n)
                end
                (dfᵖdu,dfᵖdf), fᵖ = gradient(grid.u[i], grid.fint[i], :all) do u, f
                    # normal
                    g = g⁰ + normal(u-v_rigid*Δt, n)
                    fₙ = p.penalty_force(g⋅n) * n
                    iszero(μ) && return fₙ

                    # tangential
                    if p.velocity_thresh === nothing
                        fₜ★ = tangential(-f,n) + f̄ₜ
                        fₜ = -min(1, μ*norm(fₙ)/norm(fₜ★)) * fₜ★
                    else
                        vₜ = (ūₜ + β*tangential(-f,n)/grid.m[i]*Δt^2) / Δt
                        ξ = norm(vₜ) / p.velocity_thresh
                        α = ξ<1 ? -ξ^2+2ξ : one(ξ)
                        fₜ = -α*μ*norm(fₙ)*normalize(vₜ)
                    end

                    fₙ + fₜ
                end
                grid.fᵖ[i] = fᵖ
                grid.dfᵖdu[i] = dfᵖdu
                grid.dfᵖdf[i] = dfᵖdf
            end
        end
    end
end
