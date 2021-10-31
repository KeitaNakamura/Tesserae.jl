using Poingr

function stripfooting(
        shape_function = LinearWLS(CubicBSpline());
        smooth_volumetric_strain = true,
        show_progress::Bool = true,
    )
    ρ₀ = 1.0e3
    g = 0.0
    h = 5.0
    c = 10e3
    ϕ = 0
    ψ = 0
    ν = 0.49
    E = 1e9
    dx = 0.1
    v_footing = Vec(0.0, -4.0e-3)

    grid = Grid(shape_function, 0:dx:5.0, 0:dx:5.1)
    isfooting = map(x -> x[1] ≤ 0.5 && 5.0 ≤ x[2] ≤ 5.1, grid)
    setbounds!(grid, isfooting)
    pointstate = generate_pointstate((x,y) -> y < h, grid)
    cache = MPCache(grid, pointstate.x)
    elastic = LinearElastic(; E, ν)
    model = DruckerPrager(elastic, :plane_strain; c, ϕ, ψ, tension_cutoff = Inf)

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
    output_dir = joinpath("stripfooting.tmp")
    mkpath(output_dir)

    ## paraview
    paraview_file = joinpath(output_dir, "out")
    paraview_collection(vtk_save, paraview_file)

    ## copy this file
    cp(@__FILE__, joinpath(output_dir, "main.jl"), force = true)

    logger = Logger(0.0:0.01:0.1; progress = show_progress)

    t = 0.0
    disp = Float64[]
    load = Float64[]
    while !isfinised(logger, t)

        dt = minimum(pointstate) do p
            ρ = p.m / p.V
            vc = soundspeed(elastic.K, elastic.G, ρ)
            minimum(gridsteps(grid)) / vc
        end

        update!(cache, grid, pointstate.x)
        default_point_to_grid!(grid, pointstate, cache)
        @. grid.state.v += (grid.state.f / grid.state.m) * dt

        vertical_load = 0.0
        @inbounds for bound in eachboundary(grid)
            v = grid.state.v[bound.I]
            n = bound.n
            x = grid[bound.I]
            if isfooting[bound.I] && n == Vec(0, 1)
                vertical_load += grid.state.m[bound.I] * ((v-v_footing)[2] / dt)
                v = v_footing
            elseif n == Vec(0, -1) # bottom
                v += Contact(:sticky)(v, n)
            else
                v += Contact(:slip)(v, n)
            end
            grid.state.v[bound.I] = v
        end

        default_grid_to_point!(pointstate, grid, cache, dt)

        if smooth_volumetric_strain
            Poingr.smooth_volumetric_strain!(pointstate, grid, cache, dt)
        end

        @inbounds Threads.@threads for p in eachindex(pointstate)
            ∇v = pointstate.∇v[p]
            σ_n = pointstate.σ[p]
            if smooth_volumetric_strain
                dϵ = pointstate.dϵ_v[p] / 3 * I + dev(symmetric(∇v) * dt)
            else
                dϵ = symmetric(∇v) * dt
            end
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

        push!(disp, -v_footing[2] * t)
        push!(load, vertical_load)

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
    disp, load
end
