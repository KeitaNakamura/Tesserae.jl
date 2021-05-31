using Revise, Poingr, BenchmarkTools
using Debugger

function main()
    ρ₀ = 1.0e3
    g = 9.81
    h = 5.0
    c = 10e3
    ϕ = 30
    ν = 0.3
    E = 100e7
    μ = 0.6
    it = WLS{1}(CubicBSpline(dim=2))
    grid = Grid(LinRange(0:0.1:5), LinRange(0:0.1:10))
    xₚ, V₀ₚ, hₚ = generate_pointstates((x,y) -> y < h, grid)
    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    model = DruckerPrager(LinearElastic(E = E, ν = ν), :plane_strain, c = c, ϕ = ϕ)

    mₚ = pointstate(ρ₀ * V₀ₚ)
    Vₚ = pointstate(mₚ / ρ₀)
    vₚ = pointstate(space, Vec{2,Float64})
    σₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Fₚ = pointstate(space, SecondOrderTensor{3,Float64})
    ∇vₚ = pointstate(space, SecondOrderTensor{3,Float64})
    Cₚ = pointstate(space, Mat{2,3,Float64,6})
    fill!(Fₚ, one(SecondOrderTensor{3,Float64}))

    for p in 1:npoints(space)
        σ_y = -ρ₀ * g * (h - xₚ[p][2])
        σ_x = σ_y * ν / (1 - ν)
        σₚ[p] = (@Mat [σ_x 0.0 0.0
                       0.0 σ_y 0.0
                       0.0 0.0 σ_x]) |> symmetric
    end

    fᵢ = gridstate(space, Vec{2,Float64})
    fcₙᵢ = gridstate(space, Vec{2,Float64})
    wᵢ = gridstate(space, Float64)
    mᵢ = gridstate(space, Float64)
    vᵢ = gridstate(space, Vec{2,Float64})
    v_footᵢ = gridstate(space, Vec{2,Float64})

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

    path = "results.tmp/out"
    mkpath(dirname(path))
    paraview_collection(vtk_save, path)

    foot = Rectangle(Vec(0.0, h + 0.0), Vec(0.5, h + 0.2))
    v_foot = Vec(0.0, -0.001)

    count = 0
    t = 0.0
    step = 0
    while t < 4.0
        reinit!(space, xₚ, exclude = x -> isinside(foot, x))

        ρ_min = minimum(mₚ/Vₚ)
        vc = soundspeed(model.elastic.K, model.elastic.G, ρ_min)
        dt = 0.5 * minimum(gridsteps(grid)) / vc

        mᵢ ← ∑ₚ(mₚ * N)
        wᵢ ← ∑ₚ(W)
        vᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ

        fᵢ ← ∑ₚ(-Vₚ * Tensor2D(σₚ) ⋅ ∇(N)) + ∑ₚ(mₚ * b * N)
        vᵢ ← vᵢ + (fᵢ / mᵢ) * dt

        if any(Ωc)
            fcₙ = contact_force_normal:(foot, xₚ, mₚ, vₚ, hₚ, dt, E)
            fcₙᵢ ← ∑ₚ(N * fcₙ) in Ωc
            v_footᵢ ← (∑ₚ(W * v_foot) / wᵢ) in Ωc
            fc = contact_force:(vᵢ - v_footᵢ, fcₙᵢ, mᵢ, dt, μ)
            vᵢ ← vᵢ + (fc / mᵢ) * dt
        end

        vᵢ ← (boundary_velocity:(vᵢ, nᵢ)) in dΩ

        Cₚ ← ∑ᵢ(vᵢ ⊗ (W * M⁻¹ ⋅ P(xᵢ - xₚ)))
        vₚ ← Cₚ ⋅ p0
        ∇vₚ ← Tensor3D(Cₚ ⋅ ∇p0)

        Fₚ ← Fₚ + dt*(∇vₚ ⋅ Fₚ)
        Vₚ ← V₀ₚ * det(Fₚ)
        σₚ ← stress:(model, σₚ, symmetric(∇vₚ) * dt)

        xₚ ← xₚ + vₚ * dt

        translate!(foot, v_foot * dt)
        t += dt
        step += 1

        if rem(step, 500) == 0
            paraview_collection(path, append = true) do pvd
                vtk_multiblock(string(path, count+=1)) do vtm
                    vtk_points(vtm, xₚ) do vtk
                        vtk_point_data(vtk, deviatoric_strain(infinitesimal_strain(Fₚ)), "deviatoric strain")
                    end
                    vtk_grid(vtm, foot)
                    pvd[t] = vtm
                end
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
    n = d / norm_d
    k = E * 2
    -k * (norm_d - threshold) * n
end

function contact_force(v_r::Vec, f_n::Vec, m::Real, dt::Real, μ::Real)
    iszero(f_n) && return zero(v_r)
    n = f_n / norm(f_n)
    v_t = v_r - (v_r ⋅ n) * n
    f_t = (m / dt) * v_t
    Contact(:friction, μ)(f_n + f_t, n)
end

function boundary_velocity(v::Vec, n::Vec)
    if n == Vec(0, -1) # bottom
        v + Contact(:sticky)(v, n)
    else
        v + Contact(:slip)(v, n)
    end
end
