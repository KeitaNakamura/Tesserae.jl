using Revise, Jams, BenchmarkTools
using Debugger, DelimitedFiles, Dates
using GeometricObjects

function main()
    ρ₀ = 1.0e3
    g = 9.81
    ν = 0.3
    E = 1e6
    μ = 0.3

    it = WLS{1}(CubicBSpline(dim=2))

    dx = 0.01
    grid = Grid(0:dx:0.6, 0:dx:0.6)

    θ = deg2rad(30)
    r = 0.05
    L = grid[end,1][1]
    x0 = r
    y0 = r / cos(θ) + (L - x0) * tan(θ)
    y = L*tan(θ)

    npts = 2
    xₚ, V₀ₚ, hₚ = generate_pointstates((x,y) -> (x - x0)^2 + (y - (y0 + dx/npts))^2 ≤ r^2, grid, n = npts)

    slope = Polygon([grid[1,1], grid[end,1], Vec(0.0,y)])

    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    model = LinearElastic(E = E, ν = ν)

    mₚ = pointstate(ρ₀ * V₀ₚ)
    Vₚ = pointstate(V₀ₚ)
    vₚ = pointstate(space, Vec{2,Float64})
    σₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Fₚ = pointstate(space, SecondOrderTensor{3,Float64})
    ∇vₚ = pointstate(space, SecondOrderTensor{3,Float64})
    Cₚ = pointstate(space, Mat{2,3,Float64,6})
    fill!(Fₚ, one(SecondOrderTensor{3,Float64}))

    fᵢ = gridstate(space, Vec{2,Float64})
    fcᵢ = gridstate(space, Vec{2,Float64})
    dᵢ = gridstate(space, Vec{2,Float64})
    wᵢ = gridstate(space, Float64)
    mᵢ = gridstate(space, Float64)
    vᵢ = gridstate(space, Vec{2,Float64})
    zeros!(fcᵢ)

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

    # Output files
    ## proj
    # proj_dir = joinpath(dirname(@__FILE__), "$(now()) diskrolling.tmp")
    proj_dir = joinpath(dirname(@__FILE__), "diskrolling.tmp")
    mkpath(proj_dir)

    ## paraview
    paraview_file = joinpath(proj_dir, "out")
    paraview_collection(vtk_save, paraview_file)

    ## history
    csv_file = joinpath(proj_dir, "history.csv")
    open(csv_file, "w") do io
        writedlm(io, ["t" "x_cm" "x_cm_analytical" "y_cm" "y_cm_analytical"], ',')
    end

    ## copy this file
    cp(@__FILE__, joinpath(proj_dir, basename(@__FILE__)), force = true)

    centerofmass() = sum(mₚ*xₚ)/sum(mₚ)
    x0′ = transform_coordinate(centerofmass(), Vec(x0, y0), θ)[1]

    count = 0
    t = 0.0
    step = 0
    logger = Logger(0.0:0.02:0.6; progress = true)
    while !isfinised(logger, t)
        reinit!(space, xₚ, exclude = x -> in(x, slope))

        ρ_min = minimum(mₚ/Vₚ)
        vc = soundspeed(model.K, model.G, ρ_min)
        dt = 0.5 * minimum(gridsteps(grid)) / vc

        mᵢ ← ∑ₚ(mₚ * N)
        wᵢ ← ∑ₚ(W)
        vᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ

        fᵢ ← ∑ₚ(-Vₚ * (Tensor2D(σₚ) ⋅ ∇(N))) + ∑ₚ(mₚ * b * N)
        vᵢ ← vᵢ + (fᵢ / mᵢ) * dt

        if any(Ωc)
            d = contact_distance:(slope, xₚ, mₚ, vₚ, hₚ, dt)
            dᵢ ← ∑ₚ(N * d) in Ωc
            fcᵢ ← contact_force:(vᵢ, dᵢ, mᵢ, dt, μ)
            vᵢ ← vᵢ + (fcᵢ / mᵢ) * dt
        end

        vᵢ ← (boundary_velocity:(vᵢ, nᵢ)) in dΩ

        Cₚ ← ∑ᵢ(vᵢ ⊗ (W * M⁻¹ ⋅ P(xᵢ - xₚ)))
        vₚ ← Cₚ ⋅ p0
        ∇vₚ ← Tensor3D(Cₚ ⋅ ∇p0)

        Fₚ ← Fₚ + dt*(∇vₚ ⋅ Fₚ)
        Vₚ ← V₀ₚ * det(Fₚ)
        σₚ ← stress:(model, σₚ, ∇vₚ, dt)

        xₚ ← xₚ + vₚ * dt

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
                    vtk_grid(vtm, slope)
                    pvd[t] = vtm
                end
            end

            x_cm_analytical = analytical_solution(t, θ, μ, 0, g)
            y_cm_analytical = r
            X = transform_coordinate(centerofmass(), Vec(x0, y0), θ)
            open(csv_file, "a") do io
                writedlm(io, [t r+X[1]-x0′ r+x_cm_analytical X[2]+r y_cm_analytical], ',')
            end
        end
    end
end

function analytical_solution(t::Real, θ::Real, μ::Real, x0::Real, g::Real)
    if tan(θ) > 3μ
        x0 + 1/2 * g * t^2 * (sin(θ) - μ*cos(θ))
    else
        x0 + 1/3 * g * t^2 * sin(θ)
    end
end

function transform_coordinate(x::Vec, x0::Vec, θ::Real)
    x′ = x - x0
    A = @Mat [cos(θ) -sin(θ)
              sin(θ)  cos(θ)]
    A ⋅ x′
end

function stress(model, σₚ, ∇vₚ, dt)
    σ = MaterialModels.update_stress(model, σₚ, symmetric(∇vₚ) * dt)
    MaterialModels.jaumann_stress(σ, σₚ, ∇vₚ, dt)
end

function contact_distance(poly::Polygon, x::Vec{dim, T}, m::Real, vᵣ::Vec, h::Vec{dim, T}, dt::Real) where {dim, T}
    threshold = mean(h) / 2
    d = distance(poly, x, threshold)
    d === nothing && return zero(Vec{dim, T})
    norm_d = norm(d)
    n = d / norm_d
    (threshold - norm_d) * n
end

function contact_force(v_r::Vec, d::Vec, m::Real, dt::Real, μ::Real)
    ξ = 0.96
    n = d / norm(d)
    c = m / dt
    v_n = (v_r ⋅ n) * n
    v_t = v_r - v_n
    f_n = m * (1-ξ) / dt * (d/dt + v_n)
    d_t = v_t * dt
    f_t = m * (1-ξ) / dt * 2d_t/dt
    Contact(:friction, μ, sep = true)(f_n + f_t, n)
    # Contact(:slip, sep = true)(f_n + f_t, n)

    # ξ = 0.98
    # iszero(d) && return zero(v_r)
    # f_n = -(1 - ξ) * 2m / dt^2 * d
    # n = f_n / norm(f_n)
    # v_t = v_r - (v_r ⋅ n) * n
    # f_t = (m / dt) * v_t
    # Contact(:friction, μ, sep = true)(f_n + f_t, n)
end

function boundary_velocity(v::Vec, n::Vec)
    if n == Vec(0, -1) # bottom
        v + Contact(:sticky)(v, n)
    else
        v + Contact(:slip, sep = true)(v, n)
    end
end

#=

using Plots, DelimitedFiles
plot(
    plot((arr = readdlm("examples/diskrolling.tmp/history.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,2:3])), legend = :bottomright),
    plot((arr = readdlm("examples/diskrolling.tmp/history.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,4:5])), ylims = (0.0,0.5)),
    layout = (2,1), size = (500,800)
)

=#
