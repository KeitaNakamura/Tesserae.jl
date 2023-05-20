# # Axial vibration of bar
#
# In this tutorial, we simulate axial vibration of a continuum bar based the different 
# stress update algorithms such as USL (update stress last), USF (update stress first)
# and MUSL (modified update stress last).
# For the details of calculation, see also [1], [2].
# Note that we use [`FLIP`](@ref) transfer algorithm in this example.

using Marble

# ## Stress update algorithms
#
# ### Update Stress Last (USL)
#
# USL is very common algorithm in MPM community.
# The procedure is as follows:
#
# 1. Compute grid mass $m_i^n$, momentum $m_i^n\bm{v}_i^n$ and force $\bm{f}_i^n$.
# 2. Update grid velocity (solve momentum equation on grid), then impose boundary conditions.
# 3. Compute particle velocity $\bm{v}_p^{n+1}$, velocity gradient $\nabla\bm{v}_p^{n+1}$ and position $\bm{x}_p^{n+1}$.
# 4. Update the particle stress $\bm{\sigma}_p^{n+1}$ based on the velocity gradient.
#
# Using [`particle_to_grid!`](@ref) and [`grid_to_particle!`](@ref) functions, above process can be done in Marble.jl as

function USL(grid::Grid, particles::Particles, space::MPSpace, E::Real, Δt::Real)
    fillzero!(grid)

    ## calculate grid velocity and force, update grid velocity
    particle_to_grid!((:m,:mv,:f), grid, particles, space)
    @. grid.vⁿ = grid.mv / grid.m
    @. grid.v = grid.vⁿ + Δt*(grid.f/grid.m)
    impose_boundary_condition!(grid)

    ## update particle states
    grid_to_particle!((:v,:∇v,:x), particles, grid, space, Δt)
    update_stress!.(eachparticle(particles), E, Δt)
end

# ### Update Stress First (USF)
#
# In the USF algorithm, the stress is updated before solving momentum equation on grid.
#
# 1. Compute grid mass $m_i^n$ and momentum $m_i^n\bm{v}_i^n$, then impose boundary conditions.
# 2. Compute the particle velocity gradient $\nabla\bm{v}_p^{n}$, then update the particle stress $\bm{\sigma}_p^{n}$.
# 3. Compute the grid force $\bm{f}_i^n$ based on the updated stress.
# 4. Update grid velocity (solve momentum equation on grid.), then impose boundary conditions, again.
# 5. Compute particle velocity $\bm{v}_p^{n+1}$ and position $\bm{x}_p^{n+1}$.

function USF(grid::Grid, particles::Particles, space::MPSpace, E::Real, Δt::Real)
    fillzero!(grid)

    ## calculate grid velocity
    particle_to_grid!((:m,:mv,), grid, particles, space)
    @. grid.v = grid.mv / grid.m
    impose_boundary_condition!(grid)

    ## update particle volume and stress first
    grid_to_particle!(:∇v, particles, grid, space)
    update_stress!.(eachparticle(particles), E, Δt)

    ## update grid velocity from updated stress
    particle_to_grid!(:f, grid, particles, space)
    @. grid.vⁿ = grid.v
    @. grid.v = grid.vⁿ + Δt*(grid.f/grid.m)
    impose_boundary_condition!(grid)

    ## update particle velocity and position
    grid_to_particle!((:v,:x), particles, grid, space, Δt)
end

# ### Modified Update Stress Last (MUSL)
#
# This is similar to USL, but the particle velocity gradient is calculated by
# remapped grid velocity.
#
# 1. Compute grid mass $m_i^n$, momentum $m_i^n\bm{v}_i^n$ and force $\bm{f}_i^n$.
# 2. Update grid velocity (solve momentum equation on grid), then impose boundary conditions.
# 3. Compute particle velocity $\bm{v}_p^{n+1}$ and position $\bm{x}_p^{n+1}$.
# 4. Remap grid velocity from updated particle velocity.
# 5. Compute the particle velocity gradient $\nabla\bm{v}_p^{n}$, then update the particle stress $\bm{\sigma}_p^{n}$.

function MUSL(grid::Grid, particles::Particles, space::MPSpace, E::Real, Δt::Real)
    fillzero!(grid)

    ## calculate grid velocity and force, update grid velocity
    particle_to_grid!((:m,:mv,:f), grid, particles, space)
    @. grid.vⁿ = grid.mv / grid.m
    @. grid.v = grid.vⁿ + Δt*(grid.f/grid.m)
    impose_boundary_condition!(grid)

    ## update particle velocity and position
    grid_to_particle!((:v,:x), particles, grid, space, Δt)

    ## recalculate grid velocity
    fillzero!(grid.mv)
    particle_to_grid!(:mv, grid, particles, space)
    @. grid.v = grid.mv / grid.m
    impose_boundary_condition!(grid)

    ## update particle volume and stress
    grid_to_particle!(:∇v, particles, grid, space)
    update_stress!.(eachparticle(particles), E, Δt)
end

# ## Simulation

function axial_vibration_of_bar(
        stress_update_algorithm;
        nmodes::Int,
        ncells::Int,
        PPC::Int,
        t_stop::Real,
    )

    ## simulation parameters
    n = nmodes
    v₀ = 0.1

    ## material constants
    E = 100
    ρ = 1
    L = 25
    c = √(E/ρ)

    ## exact solution
    βₙ = (2n-1)/2 * π/L
    ωₙ = βₙ * √(E/ρ)
    v(x,t) = v₀ * cos(ωₙ*t) * sin.(βₙ*x)

    ## states for grid and particles
    GridState = @NamedTuple begin
        x  :: Vec{1, Float64}
        m  :: Float64
        mv :: Vec{1, Float64}
        f  :: Vec{1, Float64}
        v  :: Vec{1, Float64}
        vⁿ :: Vec{1, Float64}
    end
    ParticleState = @NamedTuple begin
        x  :: Vec{1, Float64}
        m  :: Float64
        V  :: Float64
        v  :: Vec{1, Float64}
        ∇v :: Mat{1, 1, Float64, 1}
        σ  :: SymmetricSecondOrderTensor{1, Float64, 1}
        ϵ  :: SymmetricSecondOrderTensor{1, Float64, 1}
    end

    ## grid
    grid = generate_grid(GridState, L/ncells, (0,L))

    ## particles
    particles = generate_particles(x->true, ParticleState, grid.x; spacing=1/PPC, alg=GridSampling())
    @. particles.m = ρ * particles.V
    @. particles.v = v(particles.x, 0)

    ## create interpolation space
    space = MPSpace(LinearBSpline(), size(grid), length(particles))

    ## output
    t_list = Float64[]
    v_exa_list = Float64[]
    v_num_list = Float64[]
    Ek_list = Float64[]
    Es_list = Float64[]

    t = 0.0
    Δt = 0.1 * spacing(grid) / c
    while t ≤ t_stop

        ## update interpolation space
        update!(space, grid, particles)

        stress_update_algorithm(grid, particles, space, E, Δt)

        t += Δt

        ## store results
        v_exa = v₀/(βₙ*L) * cos(ωₙ*t)
        v_num = only(sum(@. particles.v * particles.m) / sum(particles.m))
        Ek = 1/2 * sum(@. particles.v ⋅ particles.v * particles.m)
        Es = 1/2 * sum(@. particles.σ ⊡ particles.ϵ * particles.V)
        push!(t_list, t)
        push!(v_exa_list, v_exa)
        push!(v_num_list, v_num)
        push!(Ek_list, Ek)
        push!(Es_list, Es)
    end

    (; t=t_list, v_exa=v_exa_list, v_num=v_num_list, Es=Es_list, Ek=Ek_list)
end

function impose_boundary_condition!(grid::Grid)
    grid.v[1] = zero(grid.v[1])
end

function update_stress!(pt, E::Real, Δt::Real)
    Δϵ = Δt * symmetric(pt.∇v)
    pt.V *= 1 + tr(Δϵ)
    pt.σ += E * Δϵ
    pt.ϵ += Δϵ
end

usl  = axial_vibration_of_bar(USL;  nmodes=10, ncells=50, PPC=2, t_stop=2.5)
usf  = axial_vibration_of_bar(USF;  nmodes=10, ncells=50, PPC=2, t_stop=2.5)
musl = axial_vibration_of_bar(MUSL; nmodes=10, ncells=50, PPC=2, t_stop=2.5)

import Plots
Plots.plot(usl.t, [usl.Ek usl.Es usl.Ek+usl.Es], label=["kinetic" "strain" "total"], title="USL")
Plots.plot(usf.t, [usf.Ek usf.Es usf.Ek+usf.Es], label=["kinetic" "strain" "total"], title="USF")
Plots.plot(musl.t, [musl.Ek musl.Es musl.Ek+musl.Es], label=["kinetic" "strain" "total"], title="MUSL")
# ![](USL.svg)
# ![](USF.svg)
# ![](MUSL.svg)

## check results                                                             #src
using Test                                                                   #src
if @isdefined(RUN_TESTS) && RUN_TESTS                                        #src
usl  = axial_vibration_of_bar(USL;  nmodes=1, ncells=16, PPC=2, t_stop=40.0) #src
usf  = axial_vibration_of_bar(USF;  nmodes=1, ncells=16, PPC=2, t_stop=40.0) #src
musl = axial_vibration_of_bar(MUSL; nmodes=1, ncells=16, PPC=2, t_stop=40.0) #src
@test usl.v_num ≈ usl.v_exa rtol=0.025                                       #src
@test usf.v_num ≈ usf.v_exa rtol=0.025                                       #src
@test musl.v_num ≈ musl.v_exa rtol=0.025                                     #src
end                                                                          #src

# ## References
#
# [1] Sulsky, D., Zhou, S. J., & Schreyer, H. L. (1995). Application of a particle-in-cell method to solid mechanics. *Computer physics communications*, 87(1-2), 236-252.
#
# [2] de Vaucorbeil, A., Nguyen, V. P., Sinaie, S., & Wu, J. Y. (2020). Material point method after 25 years: Theory, implementation, and applications. *Advances in applied mechanics*, 53, 185-398.
