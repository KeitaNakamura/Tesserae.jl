module SandColumn

using Poingr

struct NodeState
    f::Vec{2, Float64}
    w::Float64
    m::Float64
    v::Vec{2, Float64}
    v_n::Vec{2, Float64}
end

struct PointState
    m::Float64
    V::Float64
    x::Vec{2, Float64}
    v::Vec{2, Float64}
    b::Vec{2, Float64}
    σ::SymmetricSecondOrderTensor{3, Float64, 6}
    ϵ::SymmetricSecondOrderTensor{3, Float64, 6}
    ∇v::SecondOrderTensor{3, Float64, 9}
    C::Mat{2, 3, Float64, 6}
end

function main(; shape_function = LinearWLS(CubicBSpline()), show_progress::Bool = true)
    ρ₀ = 1.6e3
    g = 9.81
    h = 0.3
    ϕ = 38
    ψ = 0
    ν = 0.333
    E = 1e6
    dx = 0.005

    grid = Grid(NodeState, shape_function, 0:dx:1.0, 0:dx:1.0)
    pointstate = generate_pointstate((x,y) -> 0.4 < x < 0.6 && y < h, PointState, grid)
    cache = MPCache(grid, pointstate.x)
    elastic = LinearElastic(E = E, ν = ν)
    model = DruckerPrager(elastic, :plane_strain; c = 0, ϕ, ψ)

    for p in 1:length(pointstate)
        y = pointstate.x[p][2]
        σ_y = -ρ₀ * g * (h - y)
        σ_x = σ_y * ν / (1 - ν)
        pointstate.σ[p] = (@Mat [σ_x 0.0 0.0
                                 0.0 σ_y 0.0
                                 0.0 0.0 σ_x]) |> symmetric
    end
    @. pointstate.m = ρ₀ * pointstate.V
    @. pointstate.b = Vec(0.0, -g)

    @show length(pointstate)

    # Output files
    ## proj
    proj_dir = joinpath("sand.tmp")
    mkpath(proj_dir)

    ## paraview
    paraview_file = joinpath(proj_dir, "out")
    paraview_collection(vtk_save, paraview_file)

    ## copy this file
    cp(@__FILE__, joinpath(proj_dir, "main.jl"), force = true)

    logger = Logger(0.0:0.01:0.6; progress = show_progress)

    t = 0.0
    while !isfinised(logger, t)

        dt = minimum(pointstate) do p
            ρ = p.m / p.V
            vc = soundspeed(elastic.K, elastic.G, ρ)
            minimum(gridsteps(grid)) / vc
        end

        update!(cache, grid, pointstate.x)
        default_point_to_grid!(grid, pointstate, cache)
        @. grid.state.v += (grid.state.f / grid.state.m) * dt

        for bd in eachboundary(grid)
            @. grid.state.v[bd.indices] = boundary_velocity(grid.state.v[bd.indices], bd.n)
        end

        default_grid_to_point!(pointstate, grid, cache, dt)
        @inbounds Threads.@threads for p in eachindex(pointstate)
            ∇v = pointstate.∇v[p]
            σ_n = pointstate.σ[p]
            dϵ = symmetric(∇v) * dt
            σ = update_stress(model, σ_n, dϵ)
            σ = Poingr.jaumann_stress(σ, σ_n, ∇v, dt)
            if mean(σ) > model.tension_cutoff
                # In this case, since the soil particles are not contacted with
                # each other, soils should not act as continuum.
                # This means that the deformation based on the contitutitive model
                # no longer occurs.
                # So, in this process, we just calculate the elastic strain to keep
                # the consistency with the stress which is on the edge of the yield
                # function, and ignore the plastic strain to prevent excessive generation.
                # If we include this plastic strain, the volume of the material points
                # will continue to increase unexpectedly.
                σ_tr = update_stress(model.elastic, σ_n, dϵ)
                σ = Poingr.tension_cutoff(model, σ_tr)
                dϵ = elastic.Dinv ⊡ (σ - σ_n)
            end
            pointstate.σ[p] = σ
            pointstate.ϵ[p] += dϵ
            pointstate.V[p] *= exp(tr(dϵ))
        end

        update!(logger, t += dt)

        if islogpoint(logger)
            paraview_collection(paraview_file, append = true) do pvd
                vtk_multiblock(string(paraview_file, logindex(logger))) do vtm
                    vtk_points(vtm, pointstate.x) do vtk
                        ϵ = pointstate.ϵ
                        vtk["velocity"] = pointstate.v
                        vtk["mean stress"] = @dot_lazy -mean(pointstate.σ)
                        vtk["deviatoric stress"] = @dot_lazy deviatoric_stress(pointstate.σ)
                        vtk["volumetric strain"] = @dot_lazy volumetric_strain(ϵ)
                        vtk["deviatoric strain"] = @dot_lazy deviatoric_strain(ϵ)
                        vtk["stress"] = pointstate.σ
                        vtk["strain"] = ϵ
                        vtk["density"] = @dot_lazy pointstate.m / pointstate.V
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
end

function boundary_velocity(v::Vec, n::Vec)
    if n == Vec(0, -1) # bottom
        v + Contact(:friction, 0.2)(v, n)
    else
        v + Contact(:slip)(v, n)
    end
end

end
