# Getting Started

```@example
using Marble, Plots

# material constants
E = 1000                   # Young's modulus
ν = 0.3                    # Poisson's ratio
λ = (E*ν) / ((1+ν)*(1-2ν)) # Lame's first parameter
μ = E / 2(1 + ν)           # shear modulus
ρ = 1000                   # density
r = 0.2                    # radius of disk

# states for grid and particles
GridState = @NamedTuple begin
    x  :: Vec{2, Float64}
    m  :: Float64
    mv :: Vec{2, Float64}
    f  :: Vec{2, Float64}
    v  :: Vec{2, Float64}
    vⁿ :: Vec{2, Float64}
end
ParticleState = @NamedTuple begin
    x  :: Vec{2, Float64}
    m  :: Float64
    V  :: Float64
    v  :: Vec{2, Float64}
    ∇v :: SecondOrderTensor{2, Float64, 4}
    σ  :: SymmetricSecondOrderTensor{2, Float64, 3}
end

# background grid
grid = generate_grid(GridState, 0.05, (0,1), (0,1))

# particles
s = 1-r
lhs = generate_particles((x,y) -> (x-r)^2 + (y-r)^2 < r^2, ParticleState, grid) # left disk
rhs = generate_particles((x,y) -> (x-s)^2 + (y-s)^2 < r^2, ParticleState, grid) # right disk
@. lhs.m = ρ * lhs.V
@. rhs.m = ρ * rhs.V
@. lhs.v = Vec( 0.1, 0.1)
@. rhs.v = Vec(-0.1,-0.1)
particles = [lhs; rhs]

# create `LinearBSpline` interpolation space
space = MPSpace(LinearBSpline(), size(grid), length(particles))

# plot results by `Plots.@gif`
Δt = 0.001
@gif for t in range(0, 4-Δt, step=Δt)

    # update interpolation space
    update!(space, grid, particles)

    # P2G transfer
    particle_to_grid!((:m,:mv,:f), fillzero!(grid), particles, space)

    # solve momentum equation
    @. grid.vⁿ = grid.mv / grid.m
    @. grid.v  = grid.vⁿ + Δt * (grid.f/grid.m)

    # G2P transfer
    grid_to_particle!((:v,:∇v,:x), particles, grid, space, Δt)

    # update other particle states
    for p in 1:length(particles)
        Δϵ = Δt * symmetric(particles.∇v[p])
        Δσ = λ*I*tr(Δϵ) + 2μ*Δϵ
        particles.V[p] *= 1 + tr(Δϵ)
        particles.σ[p] += Δσ
    end

    # plot results
    scatter(map(Tuple, particles.x),
            lims = (0,1),
            ticks = 0:0.2:1,
            minorgrid = true,
            minorticks = 4,
            aspect_ratio = :equal,
            legend = false,)

end every 100
```
