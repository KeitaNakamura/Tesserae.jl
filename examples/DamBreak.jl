using Marble
using MaterialModels

function DamBreak(
        interp::Interpolation = LinearWLS(QuadraticBSpline()),
        transfer::Transfer    = DefaultTransfer();
        t_stop::Real          = 2.0,  # 5.0
        dx::Real              = 0.07, # 0.014
        CFL::Real             = 0.1,
        showprogress::Bool    = true,
        outdir::String        = joinpath(@__DIR__, "DamBreak.tmp"),
        output::Bool          = true,
    )

    GridState = @NamedTuple begin
        x::Vec{2, Float64}
        m::Float64
        v::Vec{2, Float64}
        vⁿ::Vec{2, Float64}
    end
    ParticleState = @NamedTuple begin
        m::Float64
        V::Float64
        r::Float64
        x::Vec{2, Float64}
        v::Vec{2, Float64}
        b::Vec{2, Float64}
        σ::SymmetricSecondOrderTensor{3, Float64, 6}
        ∇v::SecondOrderTensor{3, Float64, 9}
        B::Mat{2, 2, Float64, 4} # for APIC
        C::Mat{2, 3, Float64, 6} # for LinearWLS
    end

    g = 9.81
    ρ₀ = 1.0e3  # (kg/m3)
    μ = 1.01e-3 # (Pa⋅s)
    c = 60.0    # (m/s)

    grid = generate_grid(GridState, dx, (0,3.22), (0,4.0))
    particles = generate_particles((x,y) -> x<1.2 && y<0.6, ParticleState, grid)
    space = MPSpace(interp, grid, particles)
    model = NewtonianFluid(MorrisWaterEOS(; c, ρ_ref=ρ₀); μ)

    @. particles.σ = zero(particles.σ)
    @. particles.m = ρ₀ * particles.V
    @. particles.b = Vec(0.0, -g)

    @show length(particles)

    # Outputs
    mkpath(outdir)
    pvdfile = joinpath(outdir, "DamBreak")
    closepvd(openpvd(pvdfile))

    t = 0.0
    num_data = 100
    ts_output = collect(range(t, t_stop; length=num_data))
    while t < t_stop

        dt = CFL * minimum(LazyRows(particles)) do p
            ρ = p.m / p.V
            ν = model.μ / ρ # kinemtatic viscosity
            v = norm(p.v)
            vc = model.eos.c # speed of sound
            min(dx / (vc + v), dx^2 / ν)
        end

        update!(space, grid, particles)

        particles_to_grid!(transfer, grid, particles, space, dt)

        # boundary conditions
        slip(vᵢ, n) = vᵢ - (vᵢ⋅n)*n
        @inbounds for node in @view(LazyRows(grid)[[begin,end],:]) # left and right
            node.v = slip(node.v, Vec(1,0))
        end
        @inbounds for node in @view(LazyRows(grid)[:,[begin,end]]) # bottom and top
            node.v = slip(node.v, Vec(0,1))
        end

        grid_to_particles!(transfer, particles, grid, space, dt)

        Marble.@threaded for p in LazyRows(particles)
            m = p.m
            V = p.V
            ∇v = p.∇v

            dϵ = symmetric(dt*∇v)
            V = V * exp(tr(dϵ)) # need updated volume
            σ = @matcalc(:stress, model; d = symmetric(∇v), ρ = m/V)

            p.σ = σ
            p.V = V
        end

        t += dt

        if output && t ≥ first(ts_output)
            popfirst!(ts_output)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, num_data-length(ts_output))) do vtm
                    openvtk(vtm, particles.x) do vtk
                        vₚ = particles.v
                        σₚ = particles.σ
                        ∇vₚ = particles.∇v
                        vorticity(∇v) = ∇v[2,1] - ∇v[1,2]
                        vtk["velocity"] = vₚ
                        vtk["pressure"] = @. -mean(σₚ)
                        vtk["vorticity"] = @. vorticity(∇vₚ)
                    end
                    openvtk(vtm, grid) do vtk
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
end
