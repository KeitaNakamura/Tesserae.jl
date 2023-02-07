using Marble
using MaterialModels

function SandColumn(
        interp::Interpolation = LinearWLS(QuadraticBSpline()),
        transfer::Transfer    = Transfer(interp);
        dx::Real              = 0.01,
        CFL::Real             = 1.0,
        showprogress::Bool    = true,
        outdir::String        = joinpath(@__DIR__, "SandColumn.tmp"),
        output::Bool          = true,
    )

    GridState = @NamedTuple begin
        m::Float64
        v::Vec{2, Float64}
        vⁿ::Vec{2, Float64}
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

    ρ₀ = 1.6e3
    g = 9.81
    h = 0.3
    ϕ = deg2rad(35)
    ψ = deg2rad(0)
    ν = 0.333
    E = 1e6

    grid = Grid(0:dx:1.0, 0:dx:1.0)
    gridstate = generate_gridstate(GridState, grid)
    pointstate = generate_pointstate((x,y) -> 0.4<x<0.6 && y<h, PointState, grid)
    space = MPSpace(interp, grid, pointstate.x)
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
    pvdfile = joinpath(outdir, "SandColumn")
    closepvd(openpvd(pvdfile))

    t = 0.0
    t_stop = 0.6
    num_data = 100
    ts_output = collect(range(t, t_stop; length=num_data))
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
        @inbounds for node in @view(LazyRows(gridstate)[:,begin]) # bottom
            n = Vec(0,-1)
            μ = 0.2
            vᵢ = node.v
            v̄ₙ = vᵢ ⋅ n
            vₜ = vᵢ - v̄ₙ*n
            v̄ₜ = norm(vₜ)
            node.v = vᵢ - (v̄ₙ*n + min(μ*v̄ₙ, v̄ₜ) * (vₜ/v̄ₜ))
        end
        @inbounds for node in @view(LazyRows(gridstate)[[begin,end],:]) # left and right
            n = Vec(1,0) # this is ok for left side as well
            vᵢ = node.v
            node.v = vᵢ - (vᵢ⋅n)*n
        end

        grid_to_point!(transfer, pointstate, gridstate, space, dt)
        Marble.@threaded for p in eachindex(pointstate)
            ∇v = pointstate.∇v[p]
            σ_n = pointstate.σ[p]
            dϵ = symmetric(∇v*dt)
            ret = @matcalc(:stressall, model; σ=σ_n, dϵ)
            dσᴶ = ret.σ - σ_n
            σ = σ_n + @matcalc(:jaumann2caucy; dσ_jaumann=dσᴶ, σ=σ_n, W=skew(∇v*dt))
            if ret.status.tensioncollapse
                # recalculate strain to prevent excessive volume change
                dϵ = @matcalc(:strain, model.elastic; σ=σ-σ_n)
            end
            pointstate.σ[p] = σ
            pointstate.ϵ[p] += dϵ
            pointstate.V[p] *= exp(tr(dϵ))
        end

        t += dt

        if output && t ≥ first(ts_output)
            popfirst!(ts_output)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, num_data-length(ts_output))) do vtm
                    openvtk(vtm, pointstate.x) do vtk
                        vtk["velocity"] = pointstate.v
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
end
