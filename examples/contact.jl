using Revise, Jams, BenchmarkTools
using Debugger

function main()
    it = WLS{1}(CubicBSpline(dim=2))
    grid = Grid{2}(LinRange(0:0.02:1))
    xₚ, V₀ₚ, hₚ = generate_pointstates((x,y) -> 0.4<x<0.6 && 0.6<y<0.9, grid)
    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    model = DruckerPrager(LinearElastic(E = 1.6e5, ν = 0.3), :plane_strain, c = 0, ϕ = 30, ψ = 0.5)

    ρ₀ = 2.6e3
    mₚ = pointstate(ρ₀ * V₀ₚ)
    Vₚ = pointstate(mₚ / ρ₀)
    vₚ = pointstate(space, Vec{2,Float64})
    σₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Fₚ = pointstate(space, SecondOrderTensor{3,Float64})
    ∇vₚ = pointstate(space, SecondOrderTensor{3,Float64})
    fill!(Fₚ, one(SecondOrderTensor{3,Float64}))

    fᵢ = gridstate(space, Vec{2,Float64})
    fcᵢ = gridstate(space, Vec{2,Float64})
    mᵢ = gridstate(space, Float64)
    vᵢ = gridstate(space, Vec{2,Float64})
    vₙᵢ = gridstate(space, Vec{2,Float64})

    b = Vec(0.0, -9.81)

    N = construct(:shape_value, space)

    nᵢ = construct(:bound_normal_vector, space)
    dΩ = boundary(space)
    Ωc = nearsurface(space)

    P = polynomial(it)
    Cₚ = pointstate(space, Mat{2,3,Float64,6})
    wᵢ = gridstate(space, Float64)
    W = construct(:weight_value, space)
    M⁻¹ = construct(:moment_matrix_inverse, space)
    xᵢ = construct(:grid_coordinates, space)

    path = "results.tmp/out"
    mkpath(dirname(path))
    paraview_collection(vtk_save, path)

    poly = Rectangle(Vec(0.0, 0.4), Vec(1.0, 0.5))

    count = 0
    t = 0.0
    step = 0
    while t < 2
        reinit!(space, xₚ, exclude = x -> isinside(poly, x))

        ρ_min = minimum(mₚ/Vₚ)
        vc = soundspeed(model.elastic.K, model.elastic.G, ρ_min)
        dt = 0.5 * minimum(gridsteps(grid)) / vc

        mᵢ ← ∑ₚ(mₚ * N)
        wᵢ ← ∑ₚ(W)
        vₙᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ
        # vₙᵢ ← ∑ₚ(W * (vₚ + Tensor2D(∇vₚ) ⋅ (xᵢ - xₚ))) / wᵢ

        fᵢ ← ∑ₚ(-Vₚ * Tensor2D(σₚ) ⋅ ∇(N)) + ∑ₚ(mₚ * b * N)
        vᵢ ← vₙᵢ + (fᵢ / mᵢ) * dt

        if any(Ωc)
            # m = ∑ᵢ(mᵢ * N)
            # fc = contact_force:(poly, xₚ, m, vₚ, hₚ, dt)
            fc = contact_force:(poly, xₚ, mₚ, vₚ, hₚ, dt)
            fcᵢ ← ∑ₚ(N * fc) in Ωc
            vᵢ ← vᵢ + (fcᵢ / mᵢ) * dt
        end

        vᵢ ← (vᵢ + Contact(:friction, 0.3)(vᵢ, nᵢ)) in dΩ

        Cₚ ← ∑ᵢ(vᵢ ⊗ (W * M⁻¹ ⋅ P(xᵢ - xₚ)))
        p₀ = P(zero(Vec{2}))
        ∇p₀ = ∇(P)(zero(Vec{2}))
        vₚ ← Cₚ ⋅ p₀
        ∇vₚ ← Tensor3D(Cₚ ⋅ ∇p₀)

        Fₚ ← Fₚ + dt*(∇vₚ ⋅ Fₚ)
        Vₚ ← V₀ₚ * det(Fₚ)
        σₚ ← stress:(model, σₚ, symmetric(∇vₚ) * dt)

        xₚ ← xₚ + vₚ * dt

        t += dt
        step += 1

        if rem(step, 50) == 0
            paraview_collection(path, append = true) do pvd
                vtk_multiblock(string(path, count+=1)) do vtm
                    vtk_points(vtm, xₚ)
                    vtk_grid(vtm, poly)
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

function contact_force(poly::Polygon, x::Vec{dim, T}, m::Real, vᵣ::Vec, h::Vec{dim, T}, dt) where {dim, T}
    ξ = 0.8
    threshold = 1.0 * mean(h)
    d = distance(poly, x, threshold)
    d === nothing && return zero(Vec{dim, T})
    norm_d = norm(d)
    n = d / norm_d
    contact_force(ξ, m, vᵣ, dt, -(norm(d) - threshold), n)
end

function contact_force(ξ::Real, m::Real, vr::Vec, dt::Real, d::Real, n::Vec)
    c = m / dt
    vn = (vr ⋅ n) * n # normal
    vt = vr - vn # tangent
    fn = (1 - ξ) * ((c/dt) * d * n + c * vn) # contributions: (penetration distance) + (current normal velocity)
    ft = c * vt
    f = fn + ft
    # Contact(:friction, 0.3)(f, n)
    Contact(:sticky)(f, n)
end
