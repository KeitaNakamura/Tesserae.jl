using Marble
using MaterialModels

function StripFooting(
        interp::Interpolation = LinearWLS(QuadraticBSpline()),
        transfer::Transfer    = Transfer(interp);
        ν::Real               = 0.3,
        dx::Real              = 0.1,
        CFL::Real             = 1.0,
        lockingfree::Bool     = false,
        showprogress::Bool    = true,
        outdir::String        = joinpath(@__DIR__, "StripFooting.tmp"),
        output::Bool          = true,
    )

    GridState = @NamedTuple begin
        m::Float64
        v::Vec{2, Float64}
        vⁿ::Vec{2, Float64}
        # for smooth_pointstate!
        poly_coef::Vec{3, Float64}
        poly_mat::Mat{3, 3, Float64, 9}
    end
    PointState = @NamedTuple begin
        m::Float64
        V::Float64
        x::Vec{2, Float64}
        v::Vec{2, Float64}
        r::Vec{2, Float64}
        b::Vec{2, Float64}
        σ::SymmetricSecondOrderTensor{3, Float64, 6}
        ϵ::SymmetricSecondOrderTensor{3, Float64, 6}
        ∇v::SecondOrderTensor{3, Float64, 9}
        B::Mat{2, 2, Float64, 4} # for APIC
        C::Mat{2, 3, Float64, 6} # for LinearWLS
    end

    ρ₀ = 1.0e3
    g = 0.0
    h = 5.0
    c = 10e3
    ϕ = deg2rad(0)
    ψ = deg2rad(0)
    E = 1e9

    grid = Grid(0:dx:5.0, 0:dx:5.0)
    gridstate = generate_gridstate(GridState, grid)
    pointstate = generate_pointstate((x,y) -> y < h, PointState, grid)
    space = MPSpace(interp, grid, pointstate.x)
    elastic = LinearElastic(; E, ν)
    model = DruckerPrager(elastic, :planestrain; c, ϕ, ψ, tensioncutoff=false)

    v_footing = Vec(0.0, -4.0e-3)
    footing_indices = findall(x -> x[1]≤0.5 && x[2]==5.0, grid)

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
    pvdfile = joinpath(outdir, "StripFooting")
    closepvd(openpvd(pvdfile))

    t = 0.0
    t_stop = 0.1
    num_data = 100
    ts_output = collect(range(t, t_stop; length=num_data))
    disp = Float64[]
    load = Float64[]
    while t < t_stop

        dt = minimum(pointstate) do p
            ρ = p.m / p.V
            vc = @matcalc(:soundspeed; elastic.K, elastic.G, ρ)
            CFL * minimum(gridsteps(grid)) / vc
        end

        update!(space, pointstate)
        update_sparsity_pattern!(gridstate, space)

        point_to_grid!(transfer, gridstate, pointstate, space, dt)

        # boundary conditions
        vertical_load = 0.0
        @inbounds for i in footing_indices
            mᵢ = gridstate.m[i]
            vᵢ = gridstate.v[i]
            vertical_load += mᵢ * ((vᵢ-v_footing)[2] / dt)
            gridstate.v[i] = v_footing
        end
        # don't apply any condition (free condition) on top boundary to properly handle diriclet boundary condition
        @inbounds for node in @view(LazyRows(gridstate)[:,begin]) # bottom
            n = Vec(0,-1)
            node.v = zero(node.v)
        end
        @inbounds for node in @view(LazyRows(gridstate)[[begin,end],:]) # left and right
            n = Vec(1,0) # this is ok for left side as well
            vᵢ = node.v
            node.v = vᵢ - (vᵢ⋅n)*n
        end

        grid_to_point!(transfer, pointstate, gridstate, space, dt)

        @. tr∇v = tr(pointstate.∇v)
        if lockingfree
            Marble.smooth_pointstate!(tr∇v, pointstate.x, pointstate.V, gridstate, space)
        end

        Marble.@threaded for p in eachindex(pointstate)
            ∇v = pointstate.∇v[p]
            σ_n = pointstate.σ[p]
            if lockingfree
                ∇v_vol = @Mat [tr∇v[p]/2 0 0
                               0 tr∇v[p]/2 0
                               0 0         0]
                ∇v = ∇v_vol + dev(∇v)
            end
            dϵ = symmetric(∇v*dt)
            ret = @matcalc(:stressall, model; σ=σ_n, dϵ)
            dσᴶ = ret.σ - σ_n
            σ = σ_n + @matcalc(:jaumann2caucy; dσ_jaumann=dσᴶ, σ=σ_n, W=skew(∇v*dt))
            pointstate.σ[p] = σ
            pointstate.ϵ[p] += dϵ
            pointstate.V[p] *= exp(tr(dϵ))
        end

        t += dt

        push!(disp, -v_footing[2] * t)
        push!(load, vertical_load)

        if output && t ≥ first(ts_output)
            popfirst!(ts_output)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, num_data-length(ts_output))) do vtm
                    openvtk(vtm, pointstate.x) do vtk
                        σ = pointstate.σ
                        ϵ = pointstate.ϵ
                        vtk["mean stress"] = @. mean(σ)
                        vtk["von mises stress"] = @. sqrt(3/2 * dev(σ) ⊡ dev(σ))
                        vtk["volumetric strain"] = @. tr(ϵ)
                        vtk["deviatoric strain"] = @. sqrt(2/3 * dev(ϵ) ⊡ dev(ϵ))
                    end
                    openvtk(vtm, grid) do vtk
                    end
                    pvd[t] = vtm
                end
            end
        end
    end

    disp, load
end
