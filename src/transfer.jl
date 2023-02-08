abstract type Transfer end
struct DefaultTransfer <: Transfer end
# classical
struct FLIP  <: Transfer end
struct PIC   <: Transfer end
# affine transfer
struct AFLIP <: Transfer end
struct APIC  <: Transfer end
# Taylor transfer
struct TFLIP <: Transfer end
struct TPIC  <: Transfer end

###########
# helpers #
###########

function check_grid_particles(grid::Grid, particles::AbstractVector, space::MPSpace)
    @assert length(particles) == num_particles(space)
    @assert size(grid) == gridsize(space)
    check_grid(grid, space)
end

function check_grid(grid::Grid, space::MPSpace)
end
function check_grid(grid::SpGrid, space::MPSpace)
    if get_stamp(grid) != get_stamp(space)
        # check to use @inbounds for `SpArray`
        error("`update_sparsity_pattern!(grid, space::MPSpace)` must be executed before `particles_to_grid!` and `grid_to_particles!`")
    end
end

################
# P2G transfer #
################

# default
function particles_to_grid!(grid::Grid, particles::StructVector, space::MPSpace, dt::Real)
    particles_to_grid!(DefaultTransfer(), grid, particles, space, dt)
end

function particles_to_grid!(::Union{DefaultTransfer, FLIP, PIC}, grid::Grid, particles::StructVector, space::MPSpace, dt::Real)
    check_grid_particles(grid, particles, space)

    fillzero!(grid.m)
    fillzero!(grid.vⁿ)
    fillzero!(grid.v)

    parallel_each_particle(space) do p
        @inbounds begin
            xₚ = particles.x[p]
            mₚ = particles.m[p]
            Vₚ = particles.V[p]
            vₚ = particles.v[p]
            σₚ = particles.σ[p]
            bₚ = particles.b[p]

            Vₚσₚ = Vₚ * σₚ
            mₚbₚ = mₚ * bₚ
            mₚvₚ = mₚ * vₚ

            mp = get_mpvalue(space, p)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                f = -stress_to_force(get_system(grid), N, mp.∇N[j], xₚ, Vₚσₚ) + N*mₚbₚ
                k = unsafe_nonzeroindex(grid, i)
                grid.m[k]  += N*mₚ
                grid.vⁿ[k] += N*mₚvₚ
                grid.v[k]  += dt*f
            end
        end
    end

    @. grid.v = ((grid.vⁿ + grid.v) / grid.m) * !iszero(grid.m)
    @. grid.vⁿ = (grid.vⁿ / grid.m) * !iszero(grid.m)

    grid
end

function particles_to_grid!(::Union{AFLIP, APIC}, grid::Grid, particles::StructVector, space::MPSpace{dim, T}, dt::Real) where {dim, T}
    check_grid_particles(grid, particles, space)

    fillzero!(grid.m)
    fillzero!(grid.vⁿ)
    fillzero!(grid.v)

    parallel_each_particle(space) do p
        @inbounds begin
            xₚ = particles.x[p]
            mₚ = particles.m[p]
            Vₚ = particles.V[p]
            vₚ = particles.v[p]
            Bₚ = particles.B[p]
            σₚ = particles.σ[p]
            bₚ = particles.b[p]

            Dₚ = zero(Mat{dim, dim, T})
            mp = get_mpvalue(space, p)
            for (j, i) in enumerate(get_nodeindices(space, p))
                xᵢ = grid.x[i]
                N = mp.N[j]
                Dₚ += N*(xᵢ-xₚ)⊗(xᵢ-xₚ)
            end
            Cₚ = Bₚ ⋅ inv(Dₚ)

            Vₚσₚ = Vₚ * σₚ
            mₚbₚ = mₚ * bₚ
            mₚvₚ = mₚ * vₚ
            mₚCₚ = mₚ * Cₚ

            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                xᵢ = grid.x[i]
                f = -stress_to_force(get_system(grid), N, mp.∇N[j], xₚ, Vₚσₚ) + N*mₚbₚ
                k = unsafe_nonzeroindex(grid, i)
                grid.m[k]  += N*mₚ
                grid.vⁿ[k] += N*(mₚvₚ + mₚCₚ⋅(xᵢ-xₚ))
                grid.v[k]  += dt*f
            end
        end
    end

    @. grid.v = ((grid.vⁿ + grid.v) / grid.m) * !iszero(grid.m)
    @. grid.vⁿ = (grid.vⁿ / grid.m) * !iszero(grid.m)

    grid
end

function particles_to_grid!(::Union{TFLIP, TPIC}, grid::Grid, particles::StructVector, space::MPSpace{dim}, dt::Real) where {dim}
    check_grid_particles(grid, particles, space)

    fillzero!(grid.m)
    fillzero!(grid.vⁿ)
    fillzero!(grid.v)

    parallel_each_particle(space) do p
        @inbounds begin
            xₚ  = particles.x[p]
            mₚ  = particles.m[p]
            Vₚ  = particles.V[p]
            vₚ  = particles.v[p]
            ∇vₚ = particles.∇v[p]
            σₚ  = particles.σ[p]
            bₚ  = particles.b[p]

            Vₚσₚ = Vₚ * σₚ
            mₚbₚ = mₚ * bₚ
            mₚvₚ = mₚ * vₚ
            mₚ∇vₚ = mₚ * @Tensor(∇vₚ[1:dim, 1:dim])

            mp = get_mpvalue(space, p)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                xᵢ = grid.x[i]
                f = -stress_to_force(get_system(grid), N, mp.∇N[j], xₚ, Vₚσₚ) + N*mₚbₚ
                k = unsafe_nonzeroindex(grid, i)
                grid.m[k]  += N*mₚ
                grid.vⁿ[k] += N*(mₚvₚ + mₚ∇vₚ⋅(xᵢ-xₚ))
                grid.v[k]  += dt*f
            end
        end
    end

    @. grid.v = ((grid.vⁿ + grid.v) / grid.m) * !iszero(grid.m)
    @. grid.vⁿ = (grid.vⁿ / grid.m) * !iszero(grid.m)

    grid
end

# special default transfer for `WLS` interpolation
function particles_to_grid!(::DefaultTransfer, grid::Grid, particles::StructVector, space::MPSpace{<: Any, <: Any, <: WLS}, dt::Real)
    check_grid_particles(grid, particles, space)

    fillzero!(grid.m)
    fillzero!(grid.vⁿ)
    fillzero!(grid.v)

    parallel_each_particle(space) do p
        @inbounds begin
            xₚ = particles.x[p]
            mₚ = particles.m[p]
            Vₚ = particles.V[p]
            Cₚ = particles.C[p]
            σₚ = particles.σ[p]
            bₚ = particles.b[p]

            Vₚσₚ = Vₚ * σₚ
            mₚbₚ = mₚ * bₚ
            mₚCₚ = mₚ * Cₚ

            mp = get_mpvalue(space, p)
            P = x -> value(get_basis(mp), x)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                xᵢ = grid.x[i]
                f = -stress_to_force(get_system(grid), N, mp.∇N[j], xₚ, Vₚσₚ) + N*mₚbₚ
                k = unsafe_nonzeroindex(grid, i)
                grid.m[k] += N*mₚ
                grid.vⁿ[k] += N*mₚCₚ⋅P(xᵢ-xₚ)
                grid.v[k] += dt*f
            end
        end
    end

    @. grid.v = ((grid.vⁿ + grid.v) / grid.m) * !iszero(grid.m)
    @. grid.vⁿ = (grid.vⁿ / grid.m) * !iszero(grid.m)

    grid
end

@inline function stress_to_force(::PlaneStrain, N, ∇N, x::Vec{2}, σ::SymmetricSecondOrderTensor{3})
    Tensorial.resizedim(σ, Val(2)) ⋅ ∇N
end
@inline function stress_to_force(::Axisymmetric, N, ∇N, x::Vec{2}, σ::SymmetricSecondOrderTensor{3})
    @inbounds Tensorial.resizedim(σ, Val(2)) ⋅ ∇N + Vec(1,0) * (σ[3,3] * N / x[1])
end
@inline function stress_to_force(::NormalSystem, N, ∇N, x::Vec{3}, σ::SymmetricSecondOrderTensor{3})
    σ ⋅ ∇N
end

################
# G2P transfer #
################

# default
function grid_to_particles!(particles::StructVector, grid::Grid, space::MPSpace, dt::Real)
    grid_to_particles!(DefaultTransfer(), particles, grid, space, dt)
end

function grid_to_particles!(::Union{DefaultTransfer, FLIP, TFLIP}, particles::StructVector, grid::Grid, space::MPSpace{dim}, dt::Real) where {dim}
    check_grid_particles(grid, particles, space)

    @threaded for p in 1:num_particles(space)
        dvₚ = zero(eltype(particles.v))
        vₚ  = zero(eltype(particles.v))
        ∇vₚ = @Tensor zero(eltype(particles.∇v))[1:dim, 1:dim]
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            N = mp.N[j]
            ∇N = mp.∇N[j]
            k = unsafe_nonzeroindex(grid, i)
            vᵢ = grid.v[k]
            dvᵢ = vᵢ - grid.vⁿ[k]
            dvₚ += N * dvᵢ
            vₚ  += N * vᵢ
            ∇vₚ += vᵢ ⊗ ∇N
        end
        particles.∇v[p] = velocity_gradient(get_system(grid), particles.x[p], vₚ, ∇vₚ)
        particles.v[p] += dvₚ
        particles.x[p] += vₚ * dt
    end

    particles
end

function grid_to_particles!(::Union{PIC, TPIC}, particles::StructVector, grid::Grid, space::MPSpace{dim}, dt::Real) where {dim}
    check_grid_particles(grid, particles, space)

    @threaded for p in 1:num_particles(space)
        vₚ  = zero(eltype(particles.v))
        ∇vₚ = @Tensor zero(eltype(particles.∇v))[1:dim, 1:dim]
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            N = mp.N[j]
            ∇N = mp.∇N[j]
            k = unsafe_nonzeroindex(grid, i)
            vᵢ = grid.v[k]
            vₚ  += vᵢ * N
            ∇vₚ += vᵢ ⊗ ∇N
        end
        particles.∇v[p] = velocity_gradient(get_system(grid), particles.x[p], vₚ, ∇vₚ)
        particles.v[p] = vₚ
        particles.x[p] += vₚ * dt
    end

    particles
end

function affine_grid_to_particles!(particles::StructVector, grid::Grid, space::MPSpace, dt::Real)
    check_grid_particles(grid, particles, space)

    @threaded for p in 1:num_particles(space)
        xₚ = particles.x[p]
        Bₚ  = zero(eltype(particles.B))
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            N = mp.N[j]
            k = unsafe_nonzeroindex(grid, i)
            vᵢ = grid.v[k]
            xᵢ = grid.x[i]
            Bₚ += N * vᵢ ⊗ (xᵢ - xₚ)
        end
        particles.B[p] = Bₚ
    end

    particles
end

function grid_to_particles!(::AFLIP, particles::StructVector, grid::Grid, space::MPSpace{dim}, dt::Real) where {dim}
    affine_grid_to_particles!(particles, grid, space, dt)
    grid_to_particles!(FLIP(), particles, grid, space, dt)
    particles
end

function grid_to_particles!(::APIC, particles::StructVector, grid::Grid, space::MPSpace{dim}, dt::Real) where {dim}
    affine_grid_to_particles!(particles, grid, space, dt)
    grid_to_particles!(PIC(), particles, grid, space, dt)
    particles
end

# special default transfer for `WLS` interpolation
function grid_to_particles!(::DefaultTransfer, particles::StructVector, grid::Grid, space::MPSpace{dim, <: Any, <: WLS}, dt::Real) where {dim}
    check_grid_particles(grid, particles, space)

    @threaded for p in 1:num_particles(space)
        xₚ = particles.x[p]
        Cₚ = zero(eltype(particles.C))
        mp = get_mpvalue(space, p)
        P = x -> value(get_basis(mp), x)
        p0 = value(get_basis(mp), zero(Vec{dim, Int}))
        ∇p0 = gradient(get_basis(mp), zero(Vec{dim, Int}))
        for (j, i) in enumerate(get_nodeindices(space, p))
            w = mp.w[j]
            k = unsafe_nonzeroindex(grid, i)
            vᵢ = grid.v[k]
            xᵢ = grid.x[i]
            Minv = mp.Minv
            Cₚ += vᵢ ⊗ (w * Minv ⋅ P(xᵢ - xₚ))
        end
        vₚ = Cₚ ⋅ p0
        particles.C[p] = Cₚ
        particles.∇v[p] = velocity_gradient(get_system(grid), xₚ, vₚ, Cₚ ⋅ ∇p0)
        particles.v[p] = vₚ
        particles.x[p] += vₚ * dt
    end

    particles
end

@inline function velocity_gradient(::PlaneStrain, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    Tensorial.resizedim(∇v, Val(3)) # expaned entries are filled with zero
end
@inline function velocity_gradient(::Axisymmetric, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    @inbounds Tensorial.resizedim(∇v, Val(3)) + @Mat([0 0 0; 0 0 0; 0 0 v[1]/x[1]])
end
@inline function velocity_gradient(::NormalSystem, x::Vec{3}, v::Vec{3}, ∇v::SecondOrderTensor{3})
    ∇v
end

##########################
# smooth_particle_state! #
##########################

@generated function safe_inv(x::Mat{dim, dim, T, L}) where {dim, T, L}
    exps = fill(:z, L-1)
    quote
        @_inline_meta
        z = zero(T)
        isapproxzero(det(x)) ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x)
        # Tensorial.rank(x) != dim ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x) # this is very slow but stable
    end
end

function smooth_particle_state!(vals::AbstractVector, xₚ::AbstractVector, Vₚ::AbstractVector, grid::Grid, space::MPSpace)
    check_grid_particles(grid, vals, space)
    @assert length(vals) == length(xₚ) == length(Vₚ)

    basis = PolynomialBasis{1}()
    fillzero!(grid.poly_coef)
    fillzero!(grid.poly_mat)

    parallel_each_particle(space) do p
        @inbounds begin
            mp = get_mpvalue(space, p)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                P = value(basis, xₚ[p] - grid.x[i])
                VP = (mp.N[j] * Vₚ[p]) * P
                k = unsafe_nonzeroindex(grid, i)
                grid.poly_coef[k] += VP * vals[p]
                grid.poly_mat[k]  += VP ⊗ P
            end
        end
    end

    @. grid.poly_coef = safe_inv(grid.poly_mat) ⋅ grid.poly_coef

    @threaded for p in 1:num_particles(space)
        val = zero(eltype(vals))
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            P = value(basis, xₚ[p] - grid.x[i])
            k = unsafe_nonzeroindex(grid, i)
            val += mp.N[j] * (P ⋅ grid.poly_coef[k])
        end
        vals[p] = val
    end

    vals
end
