# Getting Started

```@example
using Sequoia
import Plots

# material constants
E = 500                    # Young's modulus
ν = 0.3                    # Poisson's ratio
λ = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
μ = E / 2(1 + ν)           # shear modulus
ρ = 1000                   # density
r = 0.2                    # radius of disk

# properties for grid and particles
GridProp = @NamedTuple begin
    x  :: Vec{2, Float64}
    m  :: Float64
    mv :: Vec{2, Float64}
    f  :: Vec{2, Float64}
    v  :: Vec{2, Float64}
    vⁿ :: Vec{2, Float64}
end
ParticleProp = @NamedTuple begin
    x  :: Vec{2, Float64}
    m  :: Float64
    V  :: Float64
    v  :: Vec{2, Float64}
    ∇v :: SecondOrderTensor{2, Float64, 4}
    σ  :: SymmetricSecondOrderTensor{2, Float64, 3}
end

# background grid
grid = generate_grid(GridProp, 0.05, (0,1), (0,1))

# particles
particles = let
    pts = generate_particles(ParticleProp, grid.x; alg=GridSampling())
    pts.V .= prod(grid.x[end]-grid.x[1]) / length(pts)

    # left disk
    lhs = filter(pts) do pt
        x, y = pt.x
        (x-r)^2 + (y-r)^2 < r^2
    end

    # right disk
    s = 1-r
    rhs = filter(pts) do pt
        x, y = pt.x
        (x-s)^2 + (y-s)^2 < r^2
    end

    lhs.v .= Vec( 0.1, 0.1)
    rhs.v .= Vec(-0.1,-0.1)
    
    [lhs; rhs]
end
@. particles.m = ρ * particles.V

# use `LinearBSpline` interpolation
mpvalues = [MPValues(Vec{2}, LinearBSpline()) for _ in 1:length(particles)]

# plot results by `Plots.@gif`
Δt = 0.001
Plots.@gif for t in range(0, 4-Δt, step=Δt)

    # update interpolation values
    for (pt, mp) in zip(particles, mpvalues)
        update!(mp, pt, grid.x)
    end

    @P2G grid=>i particles=>p mpvalues=>ip begin
        m[i]  = @∑ N[ip] * m[p]
        mv[i] = @∑ N[ip] * m[p] * v[p]
        f[i]  = @∑ -V[p] * σ[p] ⋅ ∇N[ip]
        vⁿ[i] = mv[i] / m[i]
        v[i]  = vⁿ[i] + Δt * (f[i]/m[i])
    end

    @G2P grid=>i particles=>p mpvalues=>ip begin
        v[p] += @∑ (v[i] - vⁿ[i]) * N[ip]
        ∇v[p] = @∑ v[i] ⊗ ∇N[ip]
        x[p] += @∑ Δt * v[i] * N[ip]
    end

    for p in 1:length(particles)
        Δϵ = Δt * symmetric(particles.∇v[p])
        Δσ = λ*tr(Δϵ)*I + 2μ*Δϵ
        particles.V[p] *= 1 + tr(Δϵ)
        particles.σ[p] += Δσ
    end

    # plot results
    Plots.scatter(
        reinterpret(Tuple{Float64,Float64}, particles.x),
        lims = (0,1),
        ticks = 0:0.2:1,
        minorgrid = true,
        minorticks = 4,
        aspect_ratio = :equal,
        legend = false,
    )
end every 100
```
