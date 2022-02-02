using Poingr

function sandcolumn(
        shape_function = LinearWLS(QuadraticBSpline());
        dx = 0.01,
        CFL = 1.0,
        affine_transfer::Bool = false,
        show_progress::Bool = true,
        outdir = joinpath(@__DIR__, "sandcolumn.tmp"),
    )
    ρ₀ = 1.6e3
    g = 9.81
    h = 0.3
    ϕ = 38
    ψ = 0
    ν = 0.333
    E = 1e6

    grid = Grid(shape_function, 0:dx:1.0, 0:dx:1.0)
    pointstate = generate_pointstate((x,y) -> 0.4 < x < 0.6 && y < h, grid)
    cache = MPCache(grid, pointstate.x)
    elastic = LinearElastic(; E, ν)
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

    # Outputs
    mkpath(outdir)
    paraview_file = joinpath(outdir, "out")
    Poingr.defalut_output_paraview_initialize(paraview_file)

    logger = Logger(0.0:0.01:0.6; show_progress)

    t = 0.0
    while !isfinised(logger, t)

        dt = minimum(pointstate) do p
            ρ = p.m / p.V
            vc = matcalc(Val(:sound_speed), elastic.K, elastic.G, ρ)
            CFL * minimum(gridsteps(grid)) / vc
        end

        update!(cache, grid, pointstate)
        if affine_transfer
            default_affine_point_to_grid!(grid, pointstate, cache, dt)
        else
            # use `normal` transfer function for testing purposes
            # since `default_point_to_grid!` calls `default_affine_point_to_grid!`
            # for `PolynomialBasis{1}` by default.
            Poingr.default_normal_point_to_grid!(grid, pointstate, cache, dt)
        end

        @inbounds for bound in eachboundary(grid)
            v = grid.state.v[bound.I]
            n = bound.n
            if n == Vec(0, -1) # bottom
                v += Contact(:friction, 0.2)(v, n)
            else
                v += Contact(:slip)(v, n)
            end
            grid.state.v[bound.I] = v
        end

        if affine_transfer
            default_affine_grid_to_point!(pointstate, grid, cache, dt)
        else
            # use `normal` transfer function for testing purposes
            # since `default_grid_to_point!` calls `default_affine_grid_to_point!`
            # for `PolynomialBasis{1}` by default.
            Poingr.default_normal_grid_to_point!(pointstate, grid, cache, dt)
        end
        @inbounds Threads.@threads for p in eachindex(pointstate)
            ∇v = pointstate.∇v[p]
            σ_n = pointstate.σ[p]
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

        if islogpoint(logger)
            Poingr.defalut_output_paraview_append(
                paraview_file,
                grid,
                pointstate,
                t,
                logindex(logger);
                output_grid = true,
                compress = true,
            )
        end
    end
end
