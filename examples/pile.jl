using Revise, Jams, BenchmarkTools
using Debugger, DelimitedFiles, Dates

function main()
    coord_system = :axisymmetric
    ρ₀ = 1.6e3
    g = 9.81
    h = 3
    ϕ = 38
    ν = 0.333
    E = 1e6
    μ = tan(deg2rad(28))
    # μ = 0.3
    it = WLS{1}(CubicBSpline(dim=2))
    dx = 0.015 / 3
    grid = Grid(LinRange(0:dx:0.4), LinRange(0:dx:5.0))
    xₚ, V₀ₚ, hₚ = generate_pointstates((x,y) -> y < h, grid, coord_system)
    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    elastic = LinearElastic(E = E, ν = ν)
    if coord_system == :plane_strain
        model = DruckerPrager(elastic, :plane_strain, c = 0, ϕ = ϕ, ψ = 0)
    elseif coord_system == :axisymmetric
        model = DruckerPrager(elastic, :inscribed, c = 0, ϕ = ϕ, ψ = 0)
        # model = DruckerPrager(SoilElastic(κ = 0.01, α = 40.0, p_ref = -1.0, μ_ref = 1e3), :inscribed, c = 0, ϕ = ϕ, ψ = 0)
    end

    mₚ = pointstate(ρ₀ * V₀ₚ)
    Vₚ = pointstate(mₚ / ρ₀)
    vₚ = pointstate(space, Vec{2,Float64})
    σₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Fₚ = pointstate(space, SecondOrderTensor{3,Float64})
    ∇vₚ = pointstate(space, SecondOrderTensor{3,Float64})
    Cₚ = pointstate(space, Mat{2,3,Float64,6})
    fill!(Fₚ, one(SecondOrderTensor{3,Float64}))
    fcₙₚ = pointstate(space, Vec{2,Float64})

    for p in 1:npoints(space)
        y = xₚ[p][2]
        σ_y = -ρ₀ * g * (h - y)
        σ_x = σ_y * ν / (1 - ν)
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
    vᵢ_before_contact = gridstate(space, Vec{2,Float64})
    vᵣᵢ = gridstate(space, Vec{2,Float64})
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

    thick = 2 * gridsteps(grid, 1)
    D_i = 0.15 # inner diameter at top
    d_i = 0.15 # inner diameter at tip
    taper_length = 0.715

    R_i = D_i / 2 # radius
    r_i = d_i / 2 # radius
    R_o = R_i + thick
    r_o = r_i + thick
    tip_height_0 = h + gridsteps(grid, 1)
    taper_angle = atan((R_i-r_i) / taper_length) |> rad2deg
    y_max = grid[end, end][2]
    pile = Polygon([Vec(R_i, y_max),
                    Vec(R_i, tip_height_0 + taper_length),
                    Vec(r_i, tip_height_0),
                    Vec(r_o, tip_height_0),
                    Vec(R_o, tip_height_0 + taper_length),
                    Vec(R_o, y_max)])
    v_pile = Vec(0.0, -0.1)

    pile_center_0 = center(pile)

    find_ground_pos(xₚ) = maximum(x -> x[2], filter(x -> x[1] < gridsteps(grid, 1), xₚ))
    ground_pos0 = find_ground_pos(xₚ)
    @show taper_angle

    # Output files
    ## proj
    proj_dir = joinpath(dirname(@__FILE__), "$(now()) pile.tmp")
    mkpath(proj_dir)

    ## paraview
    paraview_file = joinpath(proj_dir, "out")
    paraview_collection(vtk_save, paraview_file)

    ## history
    csv_file = joinpath(proj_dir, "history.csv")
    open(csv_file, "w") do io
        writedlm(io, ["disp" "force" "disp_inside_pile" "tip_resistance" "inside_resistance" "outside_resistance"], ',')
    end

    ## forces
    mkpath(joinpath(proj_dir, "force_tip"))
    mkpath(joinpath(proj_dir, "force_inside"))
    mkpath(joinpath(proj_dir, "force_outside"))

    ## copy this file
    cp(@__FILE__, joinpath(proj_dir, basename(@__FILE__)), force = true)

    count = 0
    t = 0.0
    step = 0
    logger = Logger(0.0:0.05:20.0; progress = true)
    while !isfinised(logger, t)
        reinit!(space, xₚ, exclude = x -> isinside(pile, x))

        ρ_min = minimum(mₚ/Vₚ)
        vc = soundspeed(model.elastic.K, model.elastic.G, ρ_min)
        dt = 0.4 * minimum(gridsteps(grid)) / vc

        mᵢ ← ∑ₚ(mₚ * N)
        wᵢ ← ∑ₚ(W)
        vᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ

        fᵢ ← ∑ₚ(-Vₚ * (stress_to_force:(coord_system, N, xₚ, σₚ))) + ∑ₚ(mₚ * b * N)
        vᵢ ← vᵢ + (fᵢ / mᵢ) * dt
        vᵢ_before_contact = vᵢ

        if any(Ωc)
            fcₙₚ ← contact_force_normal:(pile, xₚ, mₚ, vₚ, hₚ, dt, E)
            fcₙᵢ ← ∑ₚ(N * fcₙₚ) in Ωc
            vᵣᵢ ← (∑ₚ(W * (vₚ-v_pile)) / wᵢ) in Ωc
            fcᵢ ← contact_force:(vᵣᵢ, fcₙᵢ, mᵢ, dt, μ)
            vᵢ ← vᵢ + (fcᵢ / mᵢ) * dt
        end

        vᵢ ← (boundary_velocity:(vᵢ, nᵢ)) in dΩ

        Cₚ ← ∑ᵢ(vᵢ ⊗ (W * M⁻¹ ⋅ P(xᵢ - xₚ)))
        vₚ ← Cₚ ⋅ p0
        ∇vₚ ← velocity_gradient:(coord_system, xₚ, vₚ, Cₚ ⋅ ∇p0)

        Fₚ ← Fₚ + dt*(∇vₚ ⋅ Fₚ)
        Vₚ ← V₀ₚ * det(Fₚ)
        σₚ ← stress:(model, σₚ, ∇vₚ, dt)

        xₚ ← xₚ + vₚ * dt

        translate!(pile, v_pile * dt)
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
                        vtk_point_data(vtk, σₚ, "stress")
                        vtk_point_data(vtk, ϵₚ, "strain")
                        vtk_point_data(vtk, fcₙₚ, "normal contact force")
                    end
                    vtk_grid(vtm, pile)
                    vtk_grid(vtm, grid) do vtk
                        vtk_point_data(vtk, vec(vᵢ), "nodal velocity")
                        vtk_point_data(vtk, vec(vᵢ_before_contact), "nodal velocity before contact")
                        vtk_point_data(vtk, vec(vᵣᵢ), "nodal relative velocity")
                        vtk_point_data(vtk, -vec(fcₙᵢ), "nodal normal contact force")
                        vtk_point_data(vtk, vec(fcᵢ), "nodal contact force")
                    end
                    pvd[t] = vtm
                end
            end

            tip, inside, outside = extract_contact_forces(fcᵢ, fcₙᵢ, grid, pile)

            open(csv_file, "a") do io
                disp = norm(center(pile) - pile_center_0)
                force = -sum(fcᵢ)[2] * 2π
                disp_inside_pile = -(find_ground_pos(xₚ) - ground_pos0)
                writedlm(io, [disp force disp_inside_pile sum(@view tip[:,3]) sum(@view inside[:,3]) sum(@view outside[:,3])], ',')
            end

            open(io -> writedlm(io, tip, ','), joinpath(proj_dir, "force_tip", "force_tip_$(logindex(logger)).csv"), "w")
            open(io -> writedlm(io, inside, ','), joinpath(proj_dir, "force_inside", "force_inside_$(logindex(logger)).csv"), "w")
            open(io -> writedlm(io, outside, ','), joinpath(proj_dir, "force_outside", "force_outside_$(logindex(logger)).csv"), "w")
        end
    end
end

function extract_contact_forces(fcᵢ, fcₙᵢ, grid, pile)
    inside = Float64[]
    outside = Float64[]
    tip = Float64[]
    tip_height = pile[3][2]
    for I in eachindex(fcᵢ) # walk from lower height
        x = grid[I][1]
        y = grid[I][2] - tip_height
        fcy = -2π * fcᵢ[I][2]
        iszero(fcy) && continue
        if y < gridsteps(grid, 2)
            push!(tip, x)
            push!(tip, y)
            push!(tip, fcy)
        else
            line1 = Line((RigidBodies.getline(pile, 1) + reverse(RigidBodies.getline(pile, 5))) / 2)
            line2 = Line((RigidBodies.getline(pile, 2) + reverse(RigidBodies.getline(pile, 4))) / 2)
            inner = RigidBodies.ray_casting_to_right(line1, grid[I]) ||
                RigidBodies.ray_casting_to_right(line2, grid[I])
            if inner
                push!(inside, x)
                push!(inside, y)
                push!(inside, fcy)
            else
                push!(outside, x)
                push!(outside, y)
                push!(outside, fcy)
            end
        end
    end
    reshape_data = V -> reshape(V, 3, length(V)÷3)'
    map(reshape_data, (tip, inside, outside))
end

function stress(model, σₚ, ∇vₚ, dt)
    σ = MaterialModels.update_stress(model, σₚ, symmetric(∇vₚ) * dt)
    σ = MaterialModels.jaumann_stress(σ, σₚ, ∇vₚ, dt)
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
    ξ = 0.8
    -(1 - ξ) * (2m / dt^2 * (norm_d - threshold)) * n
    #=
    ξ = 0.0
    c = m / dt
    vn = (vᵣ ⋅ n) * n # normal
    vt = vᵣ - vn      # tangent
    fn = (1 - ξ) * ((c/dt) * (threshold - norm_d) * n + c * vn) # contributions: (penetration distance) + (current normal velocity)
    fn
    =#
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
plot((arr = readdlm("examples/pile.tmp/history.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,2])))
plot((arr = readdlm("examples/pile.tmp/history.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,2])), legend = false, xlims = (0,2), ylims = (0,60e3))

plot((arr = readdlm("examples/pile.tmp/history.csv", ',', skipstart=1); @show size(arr, 1); (arr[:,1], arr[:,[2,4,5,6]])), legend = false, xlims = (0,2), ylims = (0,60e3))
=#
