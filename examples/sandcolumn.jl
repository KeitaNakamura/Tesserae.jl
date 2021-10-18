using Poingr

struct NodeState
    f::Vec{2, Float64}
    w::Float64
    m::Float64
    v::Vec{2, Float64}
    v_n::Vec{2, Float64}
end

struct PointState
    m::Float64
    V0::Float64
    x::Vec{2, Float64}
    v::Vec{2, Float64}
    b::Vec{2, Float64}
    σ::SymmetricSecondOrderTensor{3, Float64, 6}
    σ0::SymmetricSecondOrderTensor{3, Float64, 6}
    F::SecondOrderTensor{3, Float64, 9}
    ∇v::SecondOrderTensor{3, Float64, 9}
    C::Mat{2, 3, Float64, 6}
end

function main()
    coord_system = PlaneStrain()

    ρ₀ = 1.6e3
    g = 9.81
    h = 0.3
    ϕ = 38
    ψ = 0
    ν = 0.333
    E = 1e6
    dx = 0.005

    grid = Grid(NodeState, LinearWLS(CubicBSpline()), 0:dx:1.0, 0:dx:1.0)
    pointstate = generate_pointstate((x,y) -> 0.4 < x < 0.6 && y < h, PointState, grid, coord_system)
    cache = MPCache(grid, pointstate.x)
    elastic = LinearElastic(E = E, ν = ν)
    model = DruckerPrager(elastic, :plane_strain; c = 0, ϕ, ψ)

    for p in 1:length(pointstate)
        y = pointstate.x[p][2]
        σ_y = -ρ₀ * g * (h - y)
        σ_x = σ_y * ν / (1 - ν)
        pointstate.σ[p] = (@Mat [σ_x 0.0 0.0
                                 0.0 σ_y 0.0
                                 0.0 0.0 σ_x]) |> symmetric
    end
    @. pointstate.m = ρ₀ * pointstate.V0
    @. pointstate.F = one(SecondOrderTensor{3,Float64})
    @. pointstate.b = Vec(0.0, -g)
    @. pointstate.σ0 = pointstate.σ

    @show length(pointstate)

    # Output files
    ## proj
    proj_dir = joinpath("sand.tmp")
    mkpath(proj_dir)

    ## paraview
    paraview_file = joinpath(proj_dir, "out")
    paraview_collection(vtk_save, paraview_file)

    ## copy this file
    cp(@__FILE__, joinpath(proj_dir, "main.jl"), force = true)

    logger = Logger(0.0:0.01:0.6; progress = true)

    t = 0.0
    while !isfinised(logger, t)

        dt = minimum(pointstate) do p
            ρ = p.m / (p.V0 * det(p.F))
            vc = soundspeed(elastic.K, elastic.G, ρ)
            minimum(gridsteps(grid)) / vc
        end

        update!(cache, grid, pointstate.x)
        default_point_to_grid!(grid, pointstate, cache, coord_system)
        @. grid.state.v += (grid.state.f / grid.state.m) * dt

        for bd in eachboundary(grid)
            @. grid.state.v[bd.indices] = boundary_velocity(grid.state.v[bd.indices], bd.n)
        end

        default_grid_to_point!(pointstate, grid, cache, dt, coord_system)
        @inbounds Threads.@threads for p in eachindex(pointstate)
            σ = update_stress(model, pointstate.σ[p], symmetric(pointstate.∇v[p]) * dt)
            σ = Poingr.jaumann_stress(σ, pointstate.σ[p], pointstate.∇v[p], dt)
            if mean(σ) > 0
                σ = zero(σ)
                # ϵv = tr(elastic.Dinv ⊡ (σ - pointstate.σ0[p]))
                # J = exp(ϵv)
                # pointstate.F[p] = J^(1/3) * one(pointstate.F[p])
            end
            pointstate.σ[p] = σ
        end

        update!(logger, t += dt)

        if islogpoint(logger)
            paraview_collection(paraview_file, append = true) do pvd
                vtk_multiblock(string(paraview_file, logindex(logger))) do vtm
                    vtk_points(vtm, pointstate.x) do vtk
                        ϵₚ = @dot_lazy symmetric(pointstate.F - $Ref(I))
                        vtk["velocity"] = pointstate.v
                        vtk["mean stress"] = @dot_lazy -mean(pointstate.σ)
                        vtk["deviatoric stress"] = @dot_lazy deviatoric_stress(pointstate.σ)
                        vtk["volumetric strain"] = @dot_lazy volumetric_strain(ϵₚ)
                        vtk["deviatoric strain"] = @dot_lazy deviatoric_strain(ϵₚ)
                        vtk["stress"] = pointstate.σ
                        vtk["strain"] = ϵₚ
                        vtk["density"] = @dot_lazy pointstate.m / (pointstate.V0 * det(pointstate.F))
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
end

function boundary_velocity(v::Vec, n::Vec)
    if n == Vec(0, -1) # bottom
        v + Contact(:friction, 0.3)(v, n)
    else
        v + Contact(:slip)(v, n)
    end
end
