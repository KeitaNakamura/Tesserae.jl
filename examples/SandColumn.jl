using Marble
using MaterialModels

function SandColumn(
        interp::Interpolation  = LinearWLS(QuadraticBSpline()),
        alg::TransferAlgorithm = DefaultTransfer();
        dx::Real               = 0.01,
        CFL::Real              = 1.0,
        showprogress::Bool     = true,
        outdir::String         = joinpath(@__DIR__, "SandColumn.tmp"),
        output::Bool           = true,
    )

    GridState = @NamedTuple begin
        x::Vec{2, Float64}
        m::Float64
        mv::Vec{2, Float64}
        f::Vec{2, Float64}
        v::Vec{2, Float64}
        vⁿ::Vec{2, Float64}
    end
    ParticleState = @NamedTuple begin
        m::Float64
        V::Float64
        l::Float64
        x::Vec{2, Float64}
        v::Vec{2, Float64}
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

    grid = generate_grid(GridState, dx, (0,1), (0,1))
    particles = generate_particles((x,y) -> 0.4<x<0.6 && y<h, ParticleState, grid)
    space = MPSpace(interp, grid, particles)
    elastic = LinearElastic(; E, ν)
    model = DruckerPrager(elastic, :planestrain; c=0, ϕ, ψ, tensioncutoff=0)

    for p in LazyRows(particles)
        y = p.x[2]
        σ_y = -ρ₀ * g * (h - y)
        σ_x = σ_y * ν / (1 - ν)
        p.σ = (@Mat [σ_x 0.0 0.0
                     0.0 σ_y 0.0
                     0.0 0.0 σ_x]) |> symmetric
    end
    @. particles.m = ρ₀ * particles.V
    @. particles.b = Vec(0.0, -g)

    @show length(particles)

    # Outputs
    mkpath(outdir)
    pvdfile = joinpath(outdir, "SandColumn")
    closepvd(openpvd(pvdfile))

    t = 0.0
    t_stop = 0.6
    num_data = 100
    ts_output = collect(range(t, t_stop; length=num_data))
    while t < t_stop

        dt = CFL * minimum(LazyRows(particles)) do p
            ρ = p.m / p.V
            vc = @matcalc(:soundspeed; elastic.K, elastic.G, ρ)
            spacing(grid) / vc
        end

        update!(space, grid, particles)

        particle_to_grid!((:m,:mv,:f), fillzero!(grid), particles, space; alg)
        @. grid.vⁿ = grid.mv / grid.m
        @. grid.v = grid.vⁿ + dt*(grid.f/grid.m)

        # boundary conditions
        @inbounds for node in @view(LazyRows(grid)[:,begin]) # bottom
            n = Vec(0,-1)
            μ = 0.2
            vᵢ = node.v
            v̄ₙ = vᵢ ⋅ n
            vₜ = vᵢ - v̄ₙ*n
            v̄ₜ = norm(vₜ)
            node.v = vᵢ - (v̄ₙ*n + min(μ*v̄ₙ/v̄ₜ, 1) * vₜ)
        end
        @inbounds for node in @view(LazyRows(grid)[[begin,end],:]) # left and right
            n = Vec(1,0) # this is ok for left side as well
            vᵢ = node.v
            node.v = vᵢ - (vᵢ⋅n)*n
        end

        grid_to_particle!((:v,:∇v,:x), particles, grid, space, dt; alg)
        Marble.@threaded for p in LazyRows(particles)
            ∇v = p.∇v
            σ_n = p.σ

            dϵ = symmetric(∇v*dt)
            ret = @matcalc(:stressall, model; σ=σ_n, dϵ)
            dσᴶ = ret.σ - σ_n
            σ = σ_n + @matcalc(:jaumann2caucy; dσ_jaumann=dσᴶ, σ=σ_n, W=skew(∇v*dt))
            if ret.status.tensioncollapse
                # recalculate strain to prevent excessive volume change
                dϵ = @matcalc(:strain, model.elastic; σ=σ-σ_n)
            end

            p.σ = σ
            p.ϵ += dϵ
            p.V *= exp(tr(dϵ))
        end

        t += dt

        if output && t ≥ first(ts_output)
            popfirst!(ts_output)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, num_data-length(ts_output))) do vtm
                    openvtk(vtm, particles.x) do vtk
                        vtk["velocity"] = particles.v
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
end
