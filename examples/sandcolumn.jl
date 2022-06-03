using Marble
using MaterialModels

function sandcolumn(
        interp = LinearWLS(QuadraticBSpline());
        dx = 0.01,
        CFL = 1.0,
        transfer = Transfer(),
        showprogress::Bool = true,
        outdir = joinpath(@__DIR__, "sandcolumn.tmp"),
    )
    ρ₀ = 1.6e3
    g = 9.81
    h = 0.3
    ϕ = deg2rad(38)
    ψ = deg2rad(0)
    ν = 0.333
    E = 1e6

    grid = Grid(0:dx:1.0, 0:dx:1.0)
    pointstate = generate_pointstate((x,y) -> 0.4 < x < 0.6 && y < h, interp, grid)
    gridstate = generate_gridstate(interp, grid)
    cache = MPCache(interp, grid, pointstate.x)
    elastic = LinearElastic(; E, ν)
    model = DruckerPrager(elastic, :planestrain; c=0, ϕ, ψ, tensioncutoff=0)

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
    Marble.defalut_output_paraview_initialize(paraview_file)

    logger = Logger(0.0, 0.6, 0.01; showprogress)

    t = 0.0
    while !isfinised(logger, t)

        dt = minimum(pointstate) do p
            ρ = p.m / p.V
            vc = @matcalc(:soundspeed; elastic.K, elastic.G, ρ)
            CFL * minimum(gridsteps(grid)) / vc
        end

        update!(cache, pointstate)
        update_sparsitypattern!(gridstate, cache)

        transfer.point_to_grid!(gridstate, pointstate, cache, dt)

        # boundary conditions
        @inbounds for (I,n) in gridbounds(grid, "-y") # bottom
            gridstate.v[I] += contacted(CoulombFriction(μ = 0.2), gridstate.v[I], n)
        end
        @inbounds for (I,n) in gridbounds(grid, "-x", "+x") # left and right
            gridstate.v[I] += contacted(CoulombFriction(μ = 0), gridstate.v[I], n)
        end

        transfer.grid_to_point!(pointstate, gridstate, cache, dt)
        @inbounds Threads.@threads for p in eachindex(pointstate)
            ∇v = pointstate.∇v[p]
            σ_n = pointstate.σ[p]
            dϵ = symmetric(∇v*dt)
            ret = @matcalc(:stressall, model; σ = σ_n, dϵ)
            dσᴶ = ret.σ - σ_n
            σ = σ_n + @matcalc(:jaumann2caucy; dσ_jaumann = dσᴶ, σ = σ_n, W = skew(∇v*dt))
            if ret.status.tensioncutoff
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

        if islogpoint(logger)
            Marble.defalut_output_paraview_append(
                paraview_file,
                grid,
                pointstate,
                t,
                logindex(logger);
                output_grid = true,
            )
        end
    end
end
