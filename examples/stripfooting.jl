using Marble
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

    grid = Grid(0:dx:5.0, 0:dx:5.0)
    pointstate = generate_pointstate((x,y) -> y < h, interp, grid)
    gridstate = generate_gridstate(interp, grid)
    space = MPSpace(interp, grid, pointstate.x)
    elastic = LinearElastic(; E, ν)
    model = DruckerPrager(elastic, :planestrain; c, ϕ, ψ, tensioncutoff=false)

    v_footing = Vec(0.0, -4.0e-3)
    footing_indices = findall(x -> x[1] ≤ 0.5 && x[2] == 5.0, grid)

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
    Marble.defalut_output_paraview_initialize(paraview_file)

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

        update!(space, pointstate)
        update_sppattern!(gridstate, space)

        transfer.point_to_grid!(gridstate, pointstate, space, dt)

        # boundary conditions
        vertical_load = 0.0
        @inbounds for I in footing_indices
            mᵢ = gridstate.m[I]
            vᵢ = gridstate.v[I]
            vertical_load += mᵢ * ((vᵢ-v_footing)[2] / dt)
            gridstate.v[I] = v_footing
        end
        # don't apply any condition (free condition) on top boundary to properly handle diriclet boundary condition
        @inbounds for (I,n) in gridbounds(grid, "-y") # bottom
            gridstate.v[I] += contacted(CoulombFriction(:sticky), gridstate.v[I], n)
        end
        @inbounds for (I,n) in gridbounds(grid, "-x", "+x") # left and right
            gridstate.v[I] += contacted(CoulombFriction(:slip), gridstate.v[I], n)
        end

        transfer.grid_to_point!(pointstate, gridstate, space, dt)

        @. tr∇v = tr(pointstate.∇v)
        if handle_volumetric_locking
            Marble.smooth_pointstate!(tr∇v, pointstate.V, gridstate, space)
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

        push!(disp, -v_footing[2] * t)
        push!(load, vertical_load)

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

    disp, load
end
