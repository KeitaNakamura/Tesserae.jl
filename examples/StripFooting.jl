using Marble
using MaterialModels

function StripFooting(
        interp::Interpolation  = LinearWLS(QuadraticBSpline()),
        alg::TransferAlgorithm = DefaultTransfer();
        ν::Real                = 0.3,
        dx::Real               = 0.1,
        CFL::Real              = 1.0,
        lockingfree::Bool      = false,
        showprogress::Bool     = true,
        outdir::String         = joinpath(@__DIR__, "StripFooting.tmp"),
        output::Bool           = true,
    )

    GridState = @NamedTuple begin
        x::Vec{2, Float64}
        m::Float64
        mv::Vec{2, Float64}
        f::Vec{2, Float64}
        v::Vec{2, Float64}
        vⁿ::Vec{2, Float64}
        # for smooth_particle_state!
        poly_coef::Vec{3, Float64}
        poly_mat::Mat{3, 3, Float64, 9}
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
        # for smooth_particle_state!
        tr∇v::Float64
    end

    ρ₀ = 1.0e3
    g = 0.0
    h = 5.0
    c = 10e3
    ϕ = deg2rad(0)
    ψ = deg2rad(0)
    E = 1e9

    grid = generate_grid(GridState, dx, (0,5), (0,5))
    particles = generate_particles((x,y) -> y < h, ParticleState, grid)
    space = MPSpace(interp, grid, particles)
    elastic = LinearElastic(; E, ν)
    model = DruckerPrager(elastic, :planestrain; c, ϕ, ψ, tensioncutoff=false)

    v_footing = Vec(0.0, -4.0e-3)
    footing_indices = findall(x -> x[1]≤0.5 && x[2]==5.0, grid.x)

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
    pvdfile = joinpath(outdir, "StripFooting")
    closepvd(openpvd(pvdfile))

    t = 0.0
    t_stop = 0.1
    num_data = 100
    ts_output = collect(range(t, t_stop; length=num_data))
    disp = Float64[]
    load = Float64[]
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
        vertical_load = 0.0
        @inbounds for i in footing_indices
            mᵢ = grid.m[i]
            vᵢ = grid.v[i]
            vertical_load += mᵢ * ((vᵢ-v_footing)[2] / dt)
            grid.v[i] = v_footing
        end
        # don't apply any condition (free condition) on top boundary to properly handle diriclet boundary condition
        @inbounds for node in @view(LazyRows(grid)[:,begin]) # bottom
            n = Vec(0,-1)
            node.v = zero(node.v)
        end
        @inbounds for node in @view(LazyRows(grid)[[begin,end],:]) # left and right
            n = Vec(1,0) # this is ok for left side as well
            vᵢ = node.v
            node.v = vᵢ - (vᵢ⋅n)*n
        end

        grid_to_particle!((:v,:∇v,:x), particles, grid, space, dt; alg)

        @. particles.tr∇v = tr(particles.∇v)
        if lockingfree
            Marble.smooth_particle_state!(particles.tr∇v, particles.x, particles.V, grid, space)
        end

        Marble.@threaded_inbounds for p in LazyRows(particles)
            ∇v = p.∇v
            σ_n = p.σ

            if lockingfree
                ∇v_vol = @Mat [p.tr∇v/2 0 0
                               0 p.tr∇v/2 0
                               0 0         0]
                ∇v = ∇v_vol + dev(∇v)
            end
            dϵ = symmetric(∇v*dt)
            ret = @matcalc(:stressall, model; σ=σ_n, dϵ)
            dσᴶ = ret.σ - σ_n
            σ = σ_n + @matcalc(:jaumann2caucy; dσ_jaumann=dσᴶ, σ=σ_n, W=skew(∇v*dt))

            p.σ = σ
            p.ϵ += dϵ
            p.V *= exp(tr(dϵ))
        end

        t += dt

        push!(disp, -v_footing[2] * t)
        push!(load, vertical_load)

        if output && t ≥ first(ts_output)
            popfirst!(ts_output)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, num_data-length(ts_output))) do vtm
                    openvtk(vtm, particles.x) do vtk
                        σ = particles.σ
                        ϵ = particles.ϵ
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
