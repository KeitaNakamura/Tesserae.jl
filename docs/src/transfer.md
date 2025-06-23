# [Transfer between grid and particles](@id manual)

## Transfer macros

```@docs
@P2G
@G2P
@G2P2G
```

## Code snippets

!!! info
    The following snippets are demonstrated in the tutorial [Transfer schemes](@ref).

In this section, the following transfer schemes are presented:

* [PIC--FLIP mixed transfer](@ref)[^1]
* [Affine PIC (APIC)](@ref)[^2]
* [Taylor PIC (TPIC)](@ref)[^3]
* [eXtended PIC (XPIC)](@ref)[^4]

[^1]: [Stomakhin, A., Schroeder, C., Chai, L., Teran, J. and Selle, A., 2013. A material point method for snow simulation. ACM Transactions on Graphics (TOG), 32(4), pp.1-10.](https://doi.org/10.1145/2461912.2461948)
[^2]: [Jiang, C., Schroeder, C., Selle, A., Teran, J. and Stomakhin, A., 2015. The affine particle-in-cell method. ACM Transactions on Graphics (TOG), 34(4), pp.1-10.](https://doi.org/10.1145/2766996)
[^3]: [Nakamura, K., Matsumura, S. and Mizutani, T., 2023. Taylor particle-in-cell transfer and kernel correction for material point method. Computer Methods in Applied Mechanics and Engineering, 403, p.115720.](https://doi.org/10.1016/j.cma.2022.115720)
[^4]: [Hammerquist, C.C. and Nairn, J.A., 2017. A new method for material point method particle updates that reduces noise and enhances stability. Computer methods in applied mechanics and engineering, 318, pp.724-738.](https://doi.org/10.1016/j.cma.2017.01.035)

We assume that the grid and particle properties have following variables:

```@example transfer
using Tesserae #hide
GridProp = @NamedTuple begin
    x   :: Vec{2, Float64} # Position
    m   :: Float64         # Mass
    mv  :: Vec{2, Float64} # Momentum
    v   :: Vec{2, Float64} # Velocity
    vⁿ  :: Vec{2, Float64} # Velocity at t = tⁿ
    # XPIC
    vᵣ★ :: Vec{2, Float64}
    v★  :: Vec{2, Float64}
end
ParticleProp = @NamedTuple begin
    x  :: Vec{2, Float64}                           # Position
    m  :: Float64                                   # Mass
    V⁰ :: Float64                                   # Initial volume
    V  :: Float64                                   # Volume
    v  :: Vec{2, Float64}                           # Velocity
    ∇v :: SecondOrderTensor{2, Float64, 4}          # Velocity gradient
    σ  :: SymmetricSecondOrderTensor{2, Float64, 3} # Cauchy stress
    F  :: SecondOrderTensor{2, Float64, 4}          # Deformation gradient
    # APIC
    B  :: SecondOrderTensor{2, Float64, 4}
    # XPIC
    vᵣ★ :: Vec{2, Float64}
    a★  :: Vec{2, Float64}
end
nothing #hide
```

```@example transfer
grid = generate_grid(GridProp, CartesianMesh(1, (0,10), (0,10)))              #hide
particles = generate_particles(ParticleProp, grid.x)                          #hide
mpvalues = generate_mpvalues(BSpline(Quadratic()), grid.x, length(particles)) #hide
for p in eachindex(particles)                                                 #hide
    update!(mpvalues[p], particles.x[p], grid.x)                              #hide
end                                                                           #hide
α = 0.5                                                                       #hide
Dₚ⁻¹ = inv(1/4 * spacing(grid.x)^2 * I)                                       #hide
m = 5                                                                         #hide
Δt = 1.0                                                                      #hide
nothing                                                                       #hide
```

### PIC--FLIP mixed transfer

```math
\begin{aligned}
m^n\bm{v}_i^n &= \sum_p w_{ip}^n m_p \bm{v}_p^n \\
\bm{v}_p^{n+1} &= \sum_i w_{ip}^n \left( (1-\alpha)\bm{v}_i^{n+1} + \alpha (\bm{v}_p^n + (\bm{v}_i^{n+1} - \bm{v}_i^n) \right)) \\
\end{aligned}
```

```@example transfer
@P2G grid=>i particles=>p mpvalues=>ip begin
    mv[i] = @∑ w[ip] * m[p] * v[p]
end
@G2P grid=>i particles=>p mpvalues=>ip begin
    v[p] = @∑ w[ip] * ((1-α)*v[i] + α*(v[p] + (v[i]-vⁿ[i])))
end
```

### Affine PIC (APIC)

```math
\begin{aligned}
m^n\bm{v}_i^n &= \sum_p w_{ip}^n m_p \left(\bm{v}_p^n + \bm{B}_p^n (\bm{D}_p^n)^{-1} (\bm{x}_i^n - \bm{x}_p^n) \right) \\
\bm{v}_p^{n+1} &= \sum_i w_{ip}^n \bm{v}_i^{n+1} \\
\bm{B}_p^{n+1} &= \sum_i w_{ip}^n \bm{v}_i^{n+1} \otimes (\bm{x}_i^n - \bm{x}_p^n)
\end{aligned}
```

```@example transfer
@P2G grid=>i particles=>p mpvalues=>ip begin
    mv[i] = @∑ w[ip] * m[p] * (v[p] + B[p] * Dₚ⁻¹ * (x[i] - x[p]))
end
@G2P grid=>i particles=>p mpvalues=>ip begin
    v[p] = @∑ w[ip] * v[i]
    B[p] = @∑ w[ip] * v[i] ⊗ (x[i] - x[p])
end
```

where `Dₚ` should be defined as `Dₚ⁻¹ = inv(1/4 * h^2 * I)` for `BSpline(Quadratic())` (see the tutorial [Transfer schemes](@ref)).

### Taylor PIC (TPIC)

```math
\begin{aligned}
m^n\bm{v}_i^n &= \sum_p w_{ip}^n m_p \left(\bm{v}_p^n + \nabla\bm{v}_p^n (\bm{x}_i^n - \bm{x}_p^n) \right) \\
\bm{v}_p^{n+1} &= \sum_i w_{ip}^n \bm{v}_i^{n+1} \\
\nabla\bm{v}_p^{n+1} &= \sum_i \bm{v}_i^{n+1} \otimes \nabla w_{ip}^n \\
\end{aligned}
```

```@example transfer
@P2G grid=>i particles=>p mpvalues=>ip begin
    mv[i] = @∑ w[ip] * m[p] * (v[p] + ∇v[p] * (x[i] - x[p]))
end
@G2P grid=>i particles=>p mpvalues=>ip begin
    v[p]  = @∑ w[ip] * v[i]
    ∇v[p] = @∑ w[ip] * v[i] ⊗ (x[i] - x[p])
end
```

### eXtended PIC (XPIC)

!!! note
    In this section, we follow the notations in the original paper[^4].

#### Overview of XPIC

We assume that $\bm{\mathsf{S}}^+$ matrix maps particle velocities to the grid and $\bm{\mathsf{S}}$ matrix maps them back.
In XPIC, a new effective acceleration for particles, $\mathbb{A}$, is employed to update the particle velocity $\bm{V}$ and position $\bm{X}$, respectively, as

```math
\begin{aligned}
\bm{V}^{(k+1)} &= \bm{V}^{(k)} + \mathbb{A}^{(k)} \Delta{t} \\
\bm{X}^{(k+1)} &= \bm{X}^{(k)} + \bm{\mathsf{S}} \bm{v}^{(k+)} \Delta{t} + \left( \frac{1}{2} \mathbb{A}^{(k)} - \bm{\mathsf{S}}\bm{a}^{(k)} \right) (\Delta{t})^2
\end{aligned}
```

where $\bm{v}^{(k+)}$ is the updated grid velocity:

```math
\bm{v}^{(k+)} = \bm{v}^{(k)} + \bm{a}^{(k)} \Delta{t}
```

The effective acceleration $\mathbb{A}$ is represented in XPIC as follows:

```math
\mathbb{A}^{(k)} \Delta{t} = (1-m) \bm{\mathsf{S}}\bm{a}^{(k)}\Delta{t} - \bm{V}^{(k)} + m\bm{\mathsf{S}}\bm{v}^{(k+)} - m\bm{\mathsf{S}}\bm{v}^{*}
```

where

```math
\bm{v}^* = \sum_r^m (-1)^r \bm{v}_r^*
```

This new $\bm{v}_r^*$ term, which is unique to XPIC($m$) with $m>1$, can be evaluated by recursion:

```math
\bm{v}_r^* = \frac{m-r+1}{r} \bm{\mathsf{S}}^+ \bm{\mathsf{S}} \bm{v}_{r-1}^*
```

starting with $\bm{v}_1^*=\bm{v}^{(k)}$.

#### Implementation using Tesserae

Based on the above equations, we can derive the following:

```math
\begin{aligned}
\bm{V}^{(k+1)} &= \bm{V}^{(k)} + \bm{\mathsf{S}}\left(\bm{v}^{(k+)}-\bm{v}^{(k)}\right) - \bm{A}^* \Delta{t} \\
\bm{X}^{(k+1)} &= \bm{X}^{(k)} + \frac{1}{2} \bm{\mathsf{S}} \left(\bm{v}^{(k+)} + \bm{v}^{(k)}\right) \Delta{t} - \frac{1}{2} \bm{A}^* (\Delta{t})^2
\end{aligned}
```

where

```math
\bm{A}^* \Delta{t} = \bm{V}^{(k)} + m \bm{\mathsf{S}} \left( \bm{v}^* - \bm{v}^{(k)} \right)
```

```@example transfer
# Set the initial values for the recursion:
@. grid.vᵣ★ = grid.vⁿ
@. grid.v★ = zero(grid.v★)

# The recursion process to calculate `v★`
for r in 2:m
    @G2P grid=>i particles=>p mpvalues=>ip begin
        vᵣ★[p] = @∑ w[ip] * vᵣ★[i]
    end
    @P2G grid=>i particles=>p mpvalues=>ip begin
        vᵣ★[i] = @∑ (m-r+1)/r * w[ip] * m[p] * vᵣ★[p] / m[i]
        v★[i] += (-1)^r * vᵣ★[i]
    end
end

# Grid-to-particle transfer in XPIC
@G2P grid=>i particles=>p mpvalues=>ip begin
    v[p] += @∑ w[ip] * (v[i] - vⁿ[i]) # same as FLIP
    x[p] += @∑ w[ip] * (v[i] + vⁿ[i]) * Δt / 2
    a★[p] = @∑ w[ip] * (v[p] + m*(v★[i] - vⁿ[i])) / Δt
    v[p] -= a★[p] * Δt
    x[p] -= a★[p] * Δt^2 / 2
end
```

where `a★` represents $\bm{A}^*$.
