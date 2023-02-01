using Marble
using MaterialModels

function DamBreak(
        interp::Interpolation = LinearWLS(QuadraticBSpline()),
        transfer::Transfer    = Transfer(interp);
        t_stop::Real          = 2.0,  # 5.0
        dx::Real              = 0.07, # 0.014
        CFL::Real             = 0.1,
        showprogress::Bool    = true,
        outdir::String        = joinpath(@__DIR__, "DamBreak.tmp"),
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
        ∇v::SecondOrderTensor{3, Float64, 9}
        B::Mat{2, 2, Float64, 4} # for APIC
        C::Mat{2, 3, Float64, 6} # for LinearWLS
    end

    g = 9.81
    ρ₀ = 1.0e3  # (kg/m3)
    μ = 1.01e-3 # (Pa⋅s)
    c = 60.0    # (m/s)

    grid = Grid(0:dx:3.22, 0:dx:4.0)
    gridstate = generate_gridstate(GridState, grid)
    pointstate = generate_pointstate((x,y) -> x<1.2 && y<0.6, PointState, grid)
    space = MPSpace(interp, grid, pointstate.x)
    model = NewtonianFluid(MorrisWaterEOS(; c, ρ_ref=ρ₀); μ)

    @. pointstate.σ = zero(pointstate.σ)
    @. pointstate.m = ρ₀ * pointstate.V
    @. pointstate.b = Vec(0.0, -g)

    @show length(pointstate)

    # Outputs
    mkpath(outdir)
    pvdfile = joinpath(outdir, "DamBreak")
    closepvd(openpvd(pvdfile))

    logger = Logger(0.0, t_stop, t_stop/100; showprogress)

    t = 0.0
    while !isfinised(logger, t)

        dt = CFL * minimum(pointstate) do pt
            ρ = pt.m / pt.V
            ν = model.μ / ρ # kinemtatic viscosity
            v = norm(pt.v)
            vc = model.eos.c # speed of sound
            min(dx / (vc + v), dx^2 / ν)
        end

        update!(space, pointstate)
        update_sparsity_pattern!(gridstate, space)

        point_to_grid!(transfer, gridstate, pointstate, space, dt)

        # boundary conditions
        @inbounds for (I,n) in gridbounds(grid, "-x", "+x", "-y", "+y")
            gridstate.v[I] += contacted(CoulombFriction(:slip), gridstate.v[I], n)
        end

        grid_to_point!(transfer, pointstate, gridstate, space, dt)

        Marble.@threaded for p in eachindex(pointstate)
            m = pointstate.m[p]
            V = pointstate.V[p]
            ∇v = pointstate.∇v[p]

            dϵ = symmetric(dt*∇v)
            V = V * exp(tr(dϵ)) # need updated volume
            σ = @matcalc(:stress, model; d = symmetric(∇v), ρ = m/V)
            pointstate.σ[p] = σ
            pointstate.V[p] = V
        end

        update!(logger, t += dt)

        if output && islogpoint(logger)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, logindex(logger))) do vtm
                    openvtk(vtm, pointstate.x) do vtk
                        vₚ = pointstate.v
                        σₚ = pointstate.σ
                        ∇vₚ = pointstate.∇v
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
