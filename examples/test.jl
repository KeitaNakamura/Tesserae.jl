using Revise, Jams, BenchmarkTools
using Debugger

function main()
    it = LinearBSpline(dim=2)
    # it = WLS{1}(QuadraticBSpline(dim=2))
    grid = CartesianGrid(0, 0.05, (20, 20))
    xₚ, V₀ₚ = generate_pointstates((x,y) -> 0.4<x<0.6 && y<0.3, grid)
    space = MPSpace(it, grid, length(xₚ))
    @show npoints(space)

    model = DruckerPrager(:plane_strain, E = 1.6e5, ν = 0.3, c = 0, ϕ = 30, ψ = 1)

    ρ₀ = 2.6e3
    mₚ = pointstate(ρ₀ * V₀ₚ)
    vₚ = pointstate(space, Vec{2,Float64})
    σₚ = pointstate(space, SymmetricSecondOrderTensor{3,Float64})
    Fₚ = pointstate(space, SecondOrderTensor{3,Float64})
    ∇vₚ = pointstate(space, SecondOrderTensor{3,Float64})
    fill!(Fₚ, one(SecondOrderTensor{3,Float64}))

    fᵢ = gridstate(space, Vec{2,Float64})
    mᵢ = gridstate(space, Float64)
    vᵢ = gridstate(space, Vec{2,Float64})
    vₙᵢ = gridstate(space, Vec{2,Float64})

    # Kᵢⱼ = gridstate_matrix(space, Vec{2,Float64})

    b = Vec(0.0, -9.81)

    # N = function_space(space, :shape_value)
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
    for step in 1:1000
        reinit!(space, xₚ)

        dt = 1e-3

        Vₚ = V₀ₚ * det(Fₚ)
        fᵢ ← ∑ₚ(-Vₚ * tensor2x2(σₚ) ⋅ ∇(N)) + ∑ₚ(mₚ * b * N)
        mᵢ ← ∑ₚ(mₚ * N)
        if it isa BSpline
            vₙᵢ ← ∑ₚ(mₚ * vₚ * N) / mᵢ
        else
            wᵢ ← ∑ₚ(W)
            vₙᵢ ← ∑ₚ(W * Cₚ ⋅ P(xᵢ - xₚ)) / wᵢ
        end

        aᵢ = fᵢ / mᵢ
        vᵢ ← vₙᵢ + aᵢ * dt

        vn = (vᵢ ⋅ nᵢ) * nᵢ
        vt = vᵢ - vn
        vᵢ ← vt in dΩ
        # dirichlet!(vᵢ, space)

        if it isa BSpline
            dvᵢ = vᵢ - vₙᵢ
            vₚ ← vₚ + ∑ᵢ(dvᵢ * N)
            ∇vₚ ← ∑ᵢ(tensor3x3(vᵢ ⊗ ∇(N)))
        else
            Cₚ ← ∑ᵢ(vᵢ ⊗ (W * M⁻¹ ⋅ P(xᵢ - xₚ)))
            p₀ = P(zero(Vec{2}))
            ∇p₀ = ∇(P)(zero(Vec{2}))
            vₚ ← Cₚ ⋅ p₀
            ∇vₚ ← tensor3x3(Cₚ ⋅ ∇p₀)
        end

        Fₚ ← Fₚ + dt*(∇vₚ ⋅ Fₚ)
        σₚ ← stress:(model, σₚ, symmetric(∇vₚ) * dt)

        if it isa BSpline
            xₚ ← xₚ + ∑ᵢ(vᵢ * N) * dt
        else
            xₚ ← xₚ + vₚ * dt
        end

        t += dt

        if rem(step, 10) == 0
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

# main()
