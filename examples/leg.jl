using Revise, Jams, BenchmarkTools
using Debugger, DelimitedFiles
using GeometricObjects

function main()
    coord_system = :axisymmetric
    ρ₀ = 0.89e3
    g = 9.81
    h = 15.0
    ϕ = 45
    ν = 0.333
    E = 10e6
    μ = 0.35
    it = WLS{1}(CubicBSpline(dim=2))
    grid = Grid(LinRange(0:0.1:25), LinRange(0:0.1:30))
    xₚ, V₀ₚ, hₚ = generate_pointstates((x,y) -> y < h, grid, coord_system)
    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    if coord_system == :plane_strain
        model = DruckerPrager(LinearElastic(E = E, ν = ν), :plane_strain, c = 0, ϕ = ϕ, ψ = 0)
    elseif coord_system == :axisymmetric
        model = DruckerPrager(LinearElastic(E = E, ν = ν), :inscribed, c = 0, ϕ = ϕ, ψ = 0)
        # model = DruckerPrager(SoilElastic(κ = 0.01, α = 40.0, p_ref = -1.0, μ_ref = 0.0[>6e3<]), :circumscribed, c = 0, ϕ = 30, ψ = 0)
    end

    mₚ = pointstate(ρ₀ * V₀ₚ)
    Vₚ = pointstate(mₚ / ρ₀)
    vₚ = pointstate(space, Vec{2,Float64})
    σₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Fₚ = pointstate(space, SecondOrderTensor{3,Float64})
    ∇vₚ = pointstate(space, SecondOrderTensor{3,Float64})
    Cₚ = pointstate(space, Mat{2,3,Float64,6})
    fill!(Fₚ, one(SecondOrderTensor{3,Float64}))

    for p in 1:npoints(space)
        y = xₚ[p][2]
        σ_y = -ρ₀ * g * (h - y)
        σ_x = 0.5 * σ_y
        σₚ[p] = (@Mat [σ_x 0.0 0.0
                       0.0 σ_y 0.0
                       0.0 0.0 σ_x]) |> symmetric
    end

    fᵢ = gridstate(space, Vec{2,Float64})
    fcᵢ = gridstate(space, Vec{2,Float64})
    fcₙᵢ = gridstate(space, Vec{2,Float64})
    wᵢ = gridstate(space, Float64)
    mᵢ = gridstate(space, Float64)
    vᵢ = gridstate(space, Vec{2,Float64})
    vᵣᵢ = gridstate(space, Vec{2,Float64})

    b = Vec(0.0, -g)

    N = construct(:shape_value, space)

    nᵢ = construct(:bound_normal_vector, space)
    dΩ = boundary(space)
    Ωc = nearsurface(space)

    P = polynomial(it)
    W = construct(:weight_value, space)
    M⁻¹ = construct(:moment_matrix_inverse, space)
    xᵢ = construct(:grid_coordinates, space)
    p0 = P(zero(Vec{2}))
    ∇p0 = ∇(P)(zero(Vec{2}))

    dy = gridsteps(grid, 2) / 4
    leg = Polygon([Vec(0.0, h+0.0), Vec(0.3375, h+0.43), Vec(1.5, h+0.72), Vec(1.5, h+0.795),
                   Vec(0.375, h+1.055), Vec(0.375, grid[1,end-1][2]), Vec(0.0, grid[1,end-1][2])] .+ Vec(0.0, dy))
    v_leg = Vec(0.0, -0.2)

    # Output files
    ## proj
    # proj_dir = joinpath(dirname(@__FILE__), "$(now()) leg.tmp")
    proj_dir = joinpath(dirname(@__FILE__), "leg.tmp")
    mkpath(proj_dir)

    ## paraview
    paraview_file = joinpath(proj_dir, "out")
    paraview_collection(vtk_save, paraview_file)

    ## history
    csv_file = joinpath(proj_dir, "history.csv")
    open(csv_file, "w") do io
        writedlm(io, ["d" "f"], ',')
    end

    ## copy this file
    cp(@__FILE__, joinpath(proj_dir, basename(@__FILE__)), force = true)

    count = 0
    t = 0.0
    step = 0
    logger = Logger(0.0:0.1:45.0; progress = true)
    while !isfinised(logger, t)
        reinit!(space, xₚ, exclude = x -> in(x, leg))

        ρ_min = minimum(mₚ/Vₚ)
        vc = soundspeed(model.elastic.K, model.elastic.G, ρ_min)
        dt = 0.5 * minimum(gridsteps(grid)) / vc

        mᵢ ← ∑ₚ(mₚ * N)
        wᵢ ← ∑ₚ(W)
        vᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ

        fᵢ ← ∑ₚ(-Vₚ * (stress_to_force:(coord_system, N, xₚ, σₚ))) + ∑ₚ(mₚ * b * N)
        vᵢ ← vᵢ + (fᵢ / mᵢ) * dt

        if any(Ωc)
            fcₙ = contact_force_normal:(leg, xₚ, mₚ, vₚ, hₚ, dt, E)
            fcₙᵢ ← ∑ₚ(N * fcₙ) in Ωc
            vᵣᵢ ← (∑ₚ(W * (vₚ - v_leg)) / wᵢ) in Ωc
            fcᵢ ← contact_force:(vᵣᵢ, fcₙᵢ, mᵢ, dt, μ)
            vᵢ ← vᵢ + (fcᵢ / mᵢ) * dt
        end

        vᵢ ← (boundary_velocity:(vᵢ, nᵢ)) in dΩ

        Cₚ ← ∑ᵢ(vᵢ ⊗ (W * M⁻¹ ⋅ P(xᵢ - xₚ)))
        vₚ ← Cₚ ⋅ p0
        ∇vₚ ← velocity_gradient:(coord_system, xₚ, vₚ, Cₚ ⋅ ∇p0)

        Fₚ ← Fₚ + dt*(∇vₚ ⋅ Fₚ)
        Vₚ ← V₀ₚ * det(Fₚ)
        σₚ ← stress:(model, σₚ, symmetric(∇vₚ) * dt)

        xₚ ← xₚ + vₚ * dt

        translate!(leg, v_leg * dt)
        update!(logger, t += dt)
        step += 1

        if islogpoint(logger)
            paraview_collection(paraview_file, append = true) do pvd
                vtk_multiblock(string(paraview_file, logindex(logger))) do vtm
                    vtk_points(vtm, xₚ) do vtk
                        ϵₚ = symmetric(Fₚ - I)
                        vtk_point_data(vtk, vₚ, "velocity")
                        vtk_point_data(vtk, -mean(σₚ), "mean stress")
                        vtk_point_data(vtk, deviatoric_stress(σₚ), "deviatoric stress")
                        vtk_point_data(vtk, volumetric_strain(ϵₚ), "volumetric strain")
                        vtk_point_data(vtk, deviatoric_strain(ϵₚ), "deviatoric strain")
                    end
                    vtk_grid(vtm, leg)
                    pvd[t] = vtm
                end
            end
            open(csv_file, "a") do io
                f = -sum(fcᵢ)[2] * 2π
                d = -v_leg[2] * t
                writedlm(io, [d f], ',')
            end
        end
    end
end

function stress(model, σₚ, dϵ)
    σ = MaterialModels.update_stress(model, σₚ, dϵ)
    mean(σ) > 0 ? zero(σ) : σ
end

function contact_force_normal(poly::Polygon, x::Vec{dim, T}, m::Real, vᵣ::Vec, h::Vec{dim, T}, dt::Real, E::Real) where {dim, T}
    threshold = mean(h) / 2
    d = distance(poly, x, threshold)
    d === nothing && return zero(Vec{dim, T})
    norm_d = norm(d)
    norm_d = norm(d)
    n = d / norm_d
    ξ = 0.8
    -(1 - ξ) * (2m / dt^2 * (norm_d - threshold)) * n
end

function contact_force(v_r::Vec, f_n::Vec, m::Real, dt::Real, μ::Real)
    iszero(f_n) && return zero(v_r)
    n = f_n / norm(f_n)
    v_t = v_r - (v_r ⋅ n) * n
    f_t = (m / dt) * v_t
    Contact(:friction, μ, sep = true)(f_n + f_t, n)
end

function boundary_velocity(v::Vec, n::Vec)
    if n == Vec(0, -1) # bottom
        v + Contact(:sticky)(v, n)
    else
        v + Contact(:slip)(v, n)
    end
end

function stress_to_force(coord_system::Symbol, N, x::Vec, σ::SymmetricSecondOrderTensor{3})
    f = Tensor2D(σ) ⋅ ∇(N)
    if coord_system == :axisymmetric
        return f + Vec(1,0) * σ[3,3] * N / x[1]
    end
    f
end

function velocity_gradient(coord_system::Symbol, x::Vec, v::Vec, ∇v::SecondOrderTensor{2})
    L = Tensor3D(∇v)
    if coord_system == :axisymmetric
        return L + @Mat [0.0 0.0 0.0
                         0.0 0.0 0.0
                         0.0 0.0 v[1] / x[1]]
    end
    L
end

#=
using Plots, DelimitedFiles
plot((arr = readdlm("examples/leg.tmp/history.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,2])), ylims = (0, 12e6), xlims = (0,2))
=#
