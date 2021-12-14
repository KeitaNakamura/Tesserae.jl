using Poingr

function stripfooting(
        shape_function = LinearWLS(CubicBSpline());
        handle_volumetric_locking = true,
        CFL = 1.0,
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
    tr∇v = @. tr(pointstate.∇v)

    @show length(pointstate)

    # Outputs
    output_dir = joinpath(@__DIR__, "stripfooting.tmp")
    paraview_file = joinpath(output_dir, "out")
    mkpath(output_dir)
    Poingr.defalut_output_paraview_initialize(paraview_file)

    logger = Logger(0.0:0.01:0.1; progress = show_progress)

    t = 0.0
    disp = Float64[]
    load = Float64[]
    while !isfinised(logger, t)

        dt = minimum(pointstate) do p
            ρ = p.m / p.V
            vc = matcalc(Val(:sound_speed), elastic.K, elastic.G, ρ)
            CFL * minimum(gridsteps(grid)) / vc
        end

        update!(cache, grid, pointstate)
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

        @. tr∇v = tr(pointstate.∇v)
        if handle_volumetric_locking
            Poingr.smooth_pointstate!(tr∇v, pointstate.V, grid, cache)
        end

        @inbounds Threads.@threads for p in eachindex(pointstate)
            ∇v = pointstate.∇v[p]
            σ_n = pointstate.σ[p]
            if handle_volumetric_locking
                ∇v_vol = @Mat [tr∇v[p]/2 0 0
                               0 tr∇v[p]/2 0
                               0 0         0]
                ∇v = ∇v_vol + dev(∇v)
            end
            dϵ = symmetric(∇v) * dt
            σ = matcalc(Val(:stress), model, σ_n, dϵ)
            σ = matcalc(Val(:jaumann_stress), σ, σ_n, ∇v, dt)
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
                σ_tr = matcalc(Val(:stress), model.elastic, σ_n, dϵ)
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
            Poingr.defalut_output_paraview_append(paraview_file, grid, pointstate, t, logindex(logger))
        end
    end

    disp, load
end
