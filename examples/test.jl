using Revise, Jams, BenchmarkTools
using Debugger

function main(; implicit = false)
    # it = LinearBSpline(dim=2)
    it = WLS{1}(QuadraticBSpline(dim=2))
    grid = Grid{2}(LinRange(0:0.05:1))
    xₚ, V₀ₚ = generate_pointstates((x,y) -> 0.4<x<0.6 && y<0.3, grid)
    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    model = DruckerPrager(:plane_strain, E = 1.6e5, ν = 0.3, c = 0, ϕ = 30, ψ = 1)

    ρ₀ = 2.6e3
    mₚ = pointstate(ρ₀ * V₀ₚ)
    Vₚ = pointstate(mₚ / ρ₀)
    vₚ = pointstate(space, Vec{2,Float64})
    σₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Fₚ = pointstate(space, SecondOrderTensor{3,Float64})
    ∇vₚ = pointstate(space, SecondOrderTensor{3,Float64})
    fill!(Fₚ, one(SecondOrderTensor{3,Float64}))

    σₙₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Dₚ = pointstate(space, SymmetricFourthOrderTensor{3,Float64})

    fᵢ = gridstate(space, Vec{2,Float64})
    mᵢ = gridstate(space, Float64)
    vᵢ = gridstate(space, Vec{2,Float64})
    vₙᵢ = gridstate(space, Vec{2,Float64})

    if implicit
        Rᵢ = gridstate(space, Vec{2,Float64})
        Kᵢⱼ = gridstate_matrix(space, Mat{2,2,Float64})
    end

    b = Vec(0.0, -9.81)

    N = construct(:shape_value, space)

    nᵢ = construct(:bound_normal_vector, space)
    dΩ = boundary(space)

    if it isa WLS
        P = polynomial(it)
        Cₚ = pointstate(space, Mat{2,3,Float64,6})
        wᵢ = gridstate(space, Float64)
        W = construct(:weight_value, space)
        M⁻¹ = construct(:moment_matrix_inverse, space)
        xᵢ = construct(:grid_coordinates, space)
    end

    path = "results.tmp/out"
    mkpath(dirname(path))
    paraview_collection(vtk_save, path)

    count = 0
    t = 0.0
    steps = implicit ? 50 : 500
    for step in 1:steps
        reinit!(space, xₚ)

        dt = implicit ? 0.01 : 0.001

        mᵢ ← ∑ₚ(mₚ * N)
        if it isa BSpline
            vₙᵢ ← ∑ₚ(mₚ * vₚ * N) / mᵢ
        else
            wᵢ ← ∑ₚ(W)
            vₙᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ
        end

        σₙₚ ← σₚ
        if implicit
            dirichlet!(vₙᵢ, space)
            vᵢ ← vₙᵢ
            reinit!(Kᵢⱼ)
            newton!(Rᵢ, Kᵢⱼ, vᵢ; maxiter=10) do Rᵢ, Kᵢⱼ, vᵢ
                ∇vₚ ← ∑ᵢ(Tensor3D(vᵢ ⊗ ∇(N)))
                Vₚ ← V₀ₚ * det(Fₚ + dt*(∇vₚ ⋅ Fₚ))

                (σₚ, Dₚ) ← stress_stiffness:(model, σₙₚ, symmetric(∇vₚ) * dt)

                fᵢ ← ∑ₚ(-Vₚ * Tensor2D(σₚ) ⋅ ∇(N)) + ∑ₚ(mₚ * b * N)
                Rᵢ ← (mᵢ / dt)  * (vᵢ - vₙᵢ) - fᵢ
                Kᵢⱼ ← ∑ₚ(dotdot(∇(N), Tensor2D(Dₚ), ∇(N)) * Vₚ * dt) + GridDiagonal(mᵢ/dt)
                dirichlet!(Rᵢ, space)
                Rᵢ, Kᵢⱼ
            end
        else
            fᵢ ← ∑ₚ(-Vₚ * Tensor2D(σₚ) ⋅ ∇(N)) + ∑ₚ(mₚ * b * N)
            aᵢ = fᵢ / mᵢ
            vᵢ ← vₙᵢ + aᵢ * dt
        end

        vn = (vᵢ ⋅ nᵢ) * nᵢ
        vt = vᵢ - vn
        # vᵢ ← vt in dΩ
        vᵢ ← (vᵢ + Contact(:slip)(vᵢ, nᵢ)) in dΩ

        if it isa BSpline
            dvᵢ = vᵢ - vₙᵢ
            vₚ ← vₚ + ∑ᵢ(dvᵢ * N)
            ∇vₚ ← ∑ᵢ(Tensor3D(vᵢ ⊗ ∇(N)))
        else
            Cₚ ← ∑ᵢ(vᵢ ⊗ (W * M⁻¹ ⋅ P(xᵢ - xₚ)))
            p₀ = P(zero(Vec{2}))
            ∇p₀ = ∇(P)(zero(Vec{2}))
            vₚ ← Cₚ ⋅ p₀
            ∇vₚ ← Tensor3D(Cₚ ⋅ ∇p₀)
        end

        Fₚ ← Fₚ + dt*(∇vₚ ⋅ Fₚ)
        Vₚ ← V₀ₚ * det(Fₚ)
        σₚ ← stress:(model, σₙₚ, symmetric(∇vₚ) * dt)

        if it isa BSpline
            xₚ ← xₚ + ∑ᵢ(vᵢ * N) * dt
        else
            xₚ ← xₚ + vₚ * dt
        end

        t += dt

        if rem(step, 5) == 0
            paraview_collection(path, append = true) do pvd
                vtk_points(string(path, count+=1), xₚ) do vtkfile
                    pvd[t] = vtkfile
                end
            end
        end
    end
end

function stress(model, σₚ, dϵ)
    σ = MaterialModels.update_stress(model, σₚ, dϵ)
    mean(σ) > 0 ? zero(σ) : σ
end

function stress_stiffness(model, σₚ, dϵ)
    D, σ = gradient(dϵ -> MaterialModels.update_stress(model, σₚ, dϵ), dϵ, :all)
    σ = mean(σ) > 0 ? zero(σ) : σ
    σ, D
end

function newton!(f!, Rᵢ, Kᵢⱼ::GridStateMatrix{<: Any, T}, xᵢ; maxiter::Int = 15, tol = sqrt(eps(T))) where {T}
    converged = false
    r = T(Inf)
    for i in 1:maxiter
        α = one(r)
        r_prev = r
        while true
            f!(Rᵢ, Kᵢⱼ, xᵢ)
            r = totalnorm(Rᵢ)
            if r < r_prev || α < 0.1
                dx = solve!(Kᵢⱼ, Rᵢ)
                xᵢ ← xᵢ - α * dx
                break
            else
                α = α^2*r_prev / 2(r + α*r_prev - r_prev)
            end
        end
        r ≤ tol && (converged = true; break)
    end
    converged
end

# main()
