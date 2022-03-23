using Poingr
using MaterialModels

function stripfooting(
        interp = LinearWLS(QuadraticBSpline());
        ν = 0.3,
        dx = 0.1,
        CFL = 1.0,
        handle_volumetric_locking::Bool = false,
        transfer = Transfer(),
        showprogress::Bool = true,
        outdir = joinpath(@__DIR__, "stripfooting.tmp"),
    )
    ρ₀ = 1.0e3
    g = 0.0
    h = 5.0
    c = 10e3
    ϕ = deg2rad(0)
    ψ = deg2rad(0)
    E = 1e9
    v_footing = Vec(0.0, -4.0e-3)

    grid = Grid(interp, 0:dx:5.0, 0:dx:5.1)
    isfooting = map(x -> x[1] ≤ 0.5 && 5.0 ≤ x[2] ≤ 5.1, grid)
    setbounds!(grid, isfooting)
    pointstate = generate_pointstate((x,y) -> y < h, grid)
    cache = MPCache(grid, pointstate.x)
    elastic = LinearElastic(; E, ν)
    model = DruckerPrager(elastic, :planestrain; c, ϕ, ψ, tensioncutoff=false)

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
    mkpath(outdir)
    paraview_file = joinpath(outdir, "out")
    Poingr.defalut_output_paraview_initialize(paraview_file)

    logger = Logger(0.0, 0.1, 0.002; showprogress)

    t = 0.0
    disp = Float64[]
    load = Float64[]
    while !isfinised(logger, t)

        dt = minimum(pointstate) do p
            ρ = p.m / p.V
            vc = @matcalc(:soundspeed; elastic.K, elastic.G, ρ)
            CFL * minimum(gridsteps(grid)) / vc
        end

        update!(cache, grid, pointstate)

        transfer.point_to_grid!(grid, pointstate, cache, dt)

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

        transfer.grid_to_point!(pointstate, grid, cache, dt)

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
            dϵ = symmetric(∇v*dt)
            ret = @matcalc(:stressall, model; σ = σ_n, dϵ)
            dσᴶ = ret.σ - σ_n
            σ = σ_n + @matcalc(:jaumann2caucy; dσ_jaumann = dσᴶ, σ = σ_n, W = skew(∇v*dt))
            if ret.status.tensioncutoff # if cutted off
                # In this case, since the soil particles are not contacted with
                # each other, soils should not act as continuum.
                # This means that the deformation based on the contitutitive model
                # no longer occurs.
                # So, in this process, we just calculate the elastic strain to keep
                # the consistency with the stress which is on the edge of the yield
                # function, and ignore the plastic strain to prevent excessive generation.
                # If we include this plastic strain, the volume of the material points
                # will continue to increase unexpectedly.
                dϵ = @matcalc(:strain, model.elastic; σ = σ - σ_n)
            end
            pointstate.σ[p] = σ
            pointstate.ϵ[p] += dϵ
            pointstate.V[p] *= exp(tr(dϵ))
        end

        update!(logger, t += dt)

        push!(disp, -v_footing[2] * t)
        push!(load, vertical_load)

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

    disp, load
end
