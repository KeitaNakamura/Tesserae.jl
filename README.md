# Tesserae.jl

*A Julia-powered toolkit for material point method*

[![CI](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/KeitaNakamura/Tesserae.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/KeitaNakamura/Tesserae.jl/graph/badge.svg?token=H5BHWIIBTG)](https://codecov.io/gh/KeitaNakamura/Tesserae.jl)

Tesserae is a toolkit for implementing the material point method in Julia. It is designed to maintain consistency between mathematical expressions and source code, enabling rapid development. Current features include:

* Grid and particle generation
* Various interpolation types, including B-splines, GIMP, CPDI, and MLS-MPM
* Convenient macros for transferring data between grid and particles
* Multi-threading support
* GPU support (CUDA and Metal)
* Exporting VTK files for visualization
* Unified framework for FEM (experimental)

![Image](https://github.com/user-attachments/assets/62acb8c4-3705-4202-b30c-fff64a690479)
![Image](https://github.com/user-attachments/assets/16379e40-9a5c-404c-9543-251e210fb00e)
![Image](https://github.com/user-attachments/assets/00bf6683-010d-4093-9cda-a5f772fe5a74)
![Image](https://github.com/user-attachments/assets/4fa4dfa2-f1cb-471e-99f1-0aa0783a2f20)

```julia
using Tesserae
import Plots

function main()
    # Parameters
    T = Float64; V2 = Vec{2,T}; M22 = Mat{2,2,T,4}
    Δt = 5e-4; E = 1e3; ν = 0.3; ρ⁰ = 1e3
    λ = (E*ν)/((1+ν)*(1-2ν)); μ = E/2(1+ν)

    # Grid, particles and interpolation
    mesh = CartesianMesh(0.01, (0,1), (0,1))
    grid = generate_grid(@NamedTuple{x::V2, m::T, v::V2, vⁿ::V2, mv::V2, f::V2}, mesh)
    pcls = generate_particles(@NamedTuple{x::V2, V::T, v::V2, ∇v::M22, F::M22, σ::M22}, mesh)
    V⁰ₚ = volume(mesh) / length(pcls); mₚ = ρ⁰ * V⁰ₚ; r = 0.2
    filter!(p -> norm(p.x.-r)<r || norm(p.x.-(1-r))<r, pcls) # Set up two disks
    @. pcls.V = V⁰ₚ; @. pcls.F = one(pcls.F)
    map!(x -> ifelse(x[1]>0.5, -0.1, 0.1) * ones(V2), pcls.v, pcls.x) # Initial velocity
    mps = generate_mpvalues(BSpline(Quadratic()), mesh, length(pcls))

    # Run simulation
    Plots.@gif for t in 0:Δt:4
        update!(mps, pcls, mesh)
        @P2G grid=>i pcls=>p mps=>ip begin
            m[i]  = @∑ w[ip] * mₚ
            mv[i] = @∑ w[ip] * mₚ * v[p]
            f[i]  = @∑ -V[p] * σ[p] * ∇w[ip]
            vⁿ[i] = mv[i] / m[i]
            v[i]  = vⁿ[i] + (f[i] / m[i]) * Δt
        end
        @G2P grid=>i pcls=>p mps=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt
            F[p] = (I + ∇v[p]*Δt) * F[p]; Jₚ = det(F[p])
            σ[p] = (μ*(F[p]*F[p]'-I) + λ*log(Jₚ)*I) / Jₚ
            V[p] = V⁰ₚ * Jₚ
        end
        # Visualize
        Plots.scatter(
            reinterpret(Tuple{T,T}, pcls.x), lims = (0,1),
            aspect_ratio = 1, markersize = 0.4, legend = false,
        )
    end every 200
end

main()
```

## Documentation

[![Stable](https://img.shields.io/badge/docs-latest%20release-blue.svg)](https://KeitaNakamura.github.io/Tesserae.jl/stable)

## Other MPM packages in Julia

* [MaterialPointSolver.jl](https://github.com/LandslideSIM/MaterialPointSolver.jl)

## Inspiration

Some functionalities are inspired from the following packages:

* [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
* [Flux.jl](https://github.com/FluxML/Flux.jl)
* [taichi_mpm](https://github.com/yuanming-hu/taichi_mpm)

## Citation

If you find Tesserae.jl useful in your work, I kindly request that you cite it as below:

```bibtex
@software{NakamuraTesserae2024,
    title = {Tesserae.jl: a {J}ulia-powered toolkit for material point method},
   author = {Nakamura, Keita},
      doi = {10.5281/zenodo.13956709},
     year = {2024},
      url = {https://github.com/KeitaNakamura/Tesserae.jl}
  licence = {MIT},
}
```
