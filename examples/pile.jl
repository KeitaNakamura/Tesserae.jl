using Revise, Jams, BenchmarkTools
using Debugger, DelimitedFiles

function main()
    coord_system = :axisymmetric
    ρ₀ = 1.6e3
    g = 9.81
    h = 3
    ϕ = 40
    ν = 0.333
    E = 10e6
    μ = 0.4
    it = WLS{1}(CubicBSpline(dim=2))
    dx = 0.015 / 2
    grid = Grid(0:dx:0.4, 0:dx:5.0)
    xₚ, V₀ₚ, hₚ = generate_pointstates((x,y) -> y < h, grid, coord_system)
    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    elastic = LinearElastic(E = E, ν = ν)
    if coord_system == :plane_strain
        model = DruckerPrager(elastic, :plane_strain, c = 0, ϕ = ϕ, ψ = 0)
    elseif coord_system == :axisymmetric
        model = DruckerPrager(elastic, :inscribed, c = 0, ϕ = ϕ, ψ = 0)
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
        σ_x = σ_y
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
    v_pileᵢ = gridstate(space, Vec{2,Float64})
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

    path = "pile.tmp/out"
    mkpath(dirname(path))
    paraview_collection(vtk_save, path)

    thick = 2 * gridsteps(grid, 1)
    D_i = 0.15 # inner diameter at top
    d_i = 0.15 # inner diameter at tip
    taper_length = 0.715

    R_i = D_i / 2 # radius
    r_i = d_i / 2 # radius
    R_o = R_i + thick
    r_o = r_i + thick
    initial_tip_height = h + gridsteps(grid, 1)
    taper_angle = atan((R_i-r_i) / taper_length) |> rad2deg
    y_max = grid[end, end][2]
    pile = Polygon([Vec(R_i, y_max),
                    Vec(R_i, initial_tip_height + taper_length),
                    Vec(r_i, initial_tip_height),
                    Vec(r_o, initial_tip_height),
                    Vec(R_o, initial_tip_height + taper_length),
                    Vec(R_o, y_max)])
    v_pile = Vec(0.0, -0.1)
    pile_pos0 = center(pile)
    @show taper_angle

    csvfile = "pile.csv"
    open(csvfile, "w") do io
        writedlm(io, ["d" "f"], ',')
    end

    count = 0
    t = 0.0
    step = 0
    logger = Logger(0.0:0.1:20.0; progress = true)
    while !isfinised(logger, t)
        reinit!(space, xₚ, exclude = x -> isinside(pile, x))

        ρ_min = minimum(mₚ/Vₚ)
        vc = soundspeed(model.elastic.K, model.elastic.G, ρ_min)
        dt = 0.5 * minimum(gridsteps(grid)) / vc

        mᵢ ← ∑ₚ(mₚ * N)
        wᵢ ← ∑ₚ(W)
        vᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ

        fᵢ ← ∑ₚ(-Vₚ * (stress_to_force:(coord_system, N, xₚ, σₚ))) + ∑ₚ(mₚ * b * N)
        vᵢ ← vᵢ + (fᵢ / mᵢ) * dt

        if any(Ωc)
            fcₙ = contact_force_normal:(pile, xₚ, mₚ, vₚ, hₚ, dt, E)
            fcₙᵢ ← ∑ₚ(N * fcₙ) in Ωc
            v_pileᵢ ← (∑ₚ(W * v_pile) / wᵢ) in Ωc
            fcᵢ ← contact_force:(vᵢ - v_pileᵢ, fcₙᵢ, mᵢ, dt, μ)
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

        translate!(pile, v_pile * dt)
        update!(logger, t += dt)
        step += 1

        if islogpoint(logger)
            paraview_collection(path, append = true) do pvd
                vtk_multiblock(string(path, logindex(logger))) do vtm
                    vtk_points(vtm, xₚ) do vtk
                        vtk_point_data(vtk, vₚ, "velocity")
                        vtk_point_data(vtk, -mean(σₚ), "mean stress")
                        vtk_point_data(vtk, deviatoric_stress(σₚ), "deviatoric stress")
                        vtk_point_data(vtk, volumetric_strain(infinitesimal_strain(Fₚ)), "volumetric strain")
                        vtk_point_data(vtk, deviatoric_strain(infinitesimal_strain(Fₚ)), "deviatoric strain")
                    end
                    vtk_grid(vtm, pile)
                    pvd[t] = vtm
                end
            end
            open(csvfile, "a") do io
                f = -sum(fcᵢ)[2] * 2π
                d = norm(center(pile) - pile_pos0)
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
    threshold = mean(h) / 2 * 1.5
    d = distance(poly, x, threshold)
    d === nothing && return zero(Vec{dim, T})
    norm_d = norm(d)
    n = d / norm_d
    k = E * 10
    # -k * (norm_d - threshold) * n
    ξ = 0.9
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
plot((arr = readdlm("pile.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,2])))
plot((arr = readdlm("pile.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,2])), legend = false, xlims = (0,2), ylims = (0,60e3))
=#
