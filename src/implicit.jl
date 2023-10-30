using LinearMaps: LinearMap

abstract type TimeIntegrationAlgorithm end

function create_default_grid_cache(grid::SpGrid{dim}) where {dim}
    Tv = eltype(grid.v)
    Tm = Mat{dim,dim,eltype(Tv),dim*dim}
    spinds = get_spinds(grid)
    StructArray(u     = SpArray{Tv}(spinds),
                δu    = SpArray{Tv}(spinds),
                R     = SpArray{Tv}(spinds),
                fint  = SpArray{Tv}(spinds),
                fext  = SpArray{Tv}(spinds),
                fᵇ    = SpArray{Tv}(spinds),
                dfᵇdu = SpArray{Tm}(spinds),
                fᵖ    = SpArray{Tv}(spinds),
                dfᵖdu = SpArray{Tm}(spinds))
end
function create_default_particles_cache(particles::Particles)
    npts = length(particles)
    T_∇u = eltype(particles.∇v)
    T_σ = eltype(particles.σ)
    T_ℂ = Tensor{Tuple{@Symmetry{3,3}, 3,3}, eltype(T_σ), 4, 54}
    StructArray(∇u=Array{T_∇u}(undef, npts), δσ=Array{T_σ}(undef, npts), ℂ=Array{T_ℂ}(undef, npts))
end
function reset_default_grid_cache!(grid::StructArray, consider_boundary_condition::Bool, consider_penalty::Bool)
    n = countnnz(get_spinds(grid.u))
    resize_nonzeros!(grid.u, n)
    resize_nonzeros!(grid.δu, n)
    resize_nonzeros!(grid.R, n)
    resize_nonzeros!(grid.fint, n)
    resize_nonzeros!(grid.fext, n)
    if consider_boundary_condition
        resize_nonzeros!(grid.fᵇ, n)
        resize_nonzeros!(grid.dfᵇdu, n)
    end
    if consider_penalty
        resize_nonzeros!(grid.fᵖ, n)
        resize_nonzeros!(grid.dfᵖdu, n)
    end
end

struct BackwardEuler <: TimeIntegrationAlgorithm end

function integration_parameters(::BackwardEuler)
    α = 1
    β = 1/2
    γ = 1
    α, β, γ
end
function create_grid_cache(grid::SpGrid, ::BackwardEuler)
    Tv = eltype(grid.v)
    spinds = get_spinds(grid)
    combine(create_default_grid_cache(grid),
            StructArray(a=SpArray{Tv}(spinds), aⁿ=SpArray{Tv}(spinds)))
end
function create_particles_cache(particles::Particles, ::BackwardEuler)
    create_default_particles_cache(particles)
end
function reset_grid_cache!(grid::StructArray, ::BackwardEuler, consider_boundary_condition::Bool, consider_penalty::Bool)
    reset_default_grid_cache!(grid, consider_boundary_condition, consider_penalty)
    n = countnnz(get_spinds(grid.u))
    resize_nonzeros!(grid.a, n)
    resize_nonzeros!(grid.aⁿ, n)
    fillzero!(grid.a)
    fillzero!(grid.aⁿ)
end

struct NewmarkBeta <: TimeIntegrationAlgorithm end

function integration_parameters(::NewmarkBeta)
    α = 1/2
    β = 1/4
    γ = 1/2
    α, β, γ
end
create_grid_cache(grid::SpGrid, ::NewmarkBeta) = create_default_grid_cache(grid)
create_particles_cache(particles::Particles, ::NewmarkBeta) = create_default_particles_cache(particles)
reset_grid_cache!(grid::StructArray, ::NewmarkBeta, consider_boundary_condition::Bool, consider_penalty::Bool) = reset_default_grid_cache!(grid, consider_boundary_condition, consider_penalty)

# utils
_tail(I::CartesianIndex) = CartesianIndex(Base.tail(Tuple(I)))

normal(v::Vec, n::Vec) = (v⋅n) * n
tangential(v::Vec, n::Vec) = v - normal(v,n)

_isfixed_bc(x::Bool) = x
_isfixed_bc(x::AbstractFloat) = x < 0
function impose_fixed_boundary_condition!(gridstates::Tuple{Vararg{AbstractArray{<: Vec{dim}, dim}}}, bc::AbstractArray{<: Real}) where {dim}
    @assert all(==(size(first(gridstates))), map(size, gridstates))
    @assert size(bc) == (dim, size(first(gridstates))...)
    for I in CartesianIndices(bc)
        if _isfixed_bc(bc[I])
            for gridstate in gridstates
                flatarray(gridstate)[I] = 0
            end
        end
    end
end

function compute_flatfreeindices(grid::Grid{dim}, isfixed::AbstractArray{Bool}) where {dim}
    @assert size(isfixed) == (dim, size(grid)...)
    filter(CartesianIndices(isfixed)) do I
        I′ = _tail(I)
        @inbounds isactive(grid, I′) && !isfixed[I]
    end
end
function compute_flatfreeindices(grid::Grid{dim}, coefs::AbstractArray{<: AbstractFloat}) where {dim}
    @assert size(coefs) == (dim, size(grid)...)
    filter(CartesianIndices(coefs)) do I
        I′ = _tail(I)
        @inbounds isactive(grid, I′) && coefs[I] ≥ 0 # negative friction coefficient means fixed
    end
end

function compute_boundary_friction!(grid::Grid, Δt::Real, coefs::AbstractArray{<: AbstractFloat})
    fillzero!(grid.fᵇ)
    fillzero!(grid.dfᵇdv)
    normals = NormalVectorArray(size(grid))
    for I in CartesianIndices(coefs)
        μ = coefs[I]
        I′ = _tail(I)
        if μ > 0 && isactive(grid, I′)
            i = nonzeroindex(grid, I′)
            n = normals[i]
            fint = grid.fint[i]
            if fint ⋅ n < 0
                dfᵇdv, fᵇ = gradient(grid.v[i], :all) do v
                    fₙ = normal(f,n)
                    vₜ = tangential(v,n)
                    f̄ₜ = grid.m[i] * vₜ / Δt
                    -min(1, μ*norm(fₙ)/norm(f̄ₜ)) * f̄ₜ
                end
                grid.fᵇ[i] += fᵇ
                grid.dfᵇdv[i] += dfᵇdv
            end
        end
    end
end

struct ImplicitIntegrator{Alg <: TimeIntegrationAlgorithm, T}
    alg::Alg
    α::T
    β::T
    γ::T
    nlsolver::NonlinearSolver
    linsolve!::Function
    grid_cache::StructArray
    particles_cache::StructVector
end

function ImplicitIntegrator(
                  :: Type{T},
        alg       :: TimeIntegrationAlgorithm,
        grid      :: Grid,
        particles :: Particles;
        abstol    :: Real = sqrt(eps(T)),
        reltol    :: Real = zero(T),
        maxiter   :: Int  = 100,
        linsolve!         = (x,A,b) -> idrs!(x,A,b),
    ) where {T}
    α, β, γ = integration_parameters(alg)
    nlsolver = NewtonSolver(T; abstol, reltol, maxiter)
    grid_cache = create_grid_cache(grid, alg)
    particles_cache = fillzero!(create_particles_cache(particles, alg))
    ImplicitIntegrator{typeof(alg), T}(alg, α, β, γ, nlsolver, linsolve!, grid_cache, particles_cache)
end
ImplicitIntegrator(alg::TimeIntegrationAlgorithm, grid::Grid, particles::Particles; kwargs...) = ImplicitIntegrator(Float64, alg, grid, particles; kwargs...)

function solve_grid_velocity!(
        update_stress! :: Any,
        grid           :: Grid{dim},
        particles      :: Particles,
        space          :: MPSpace{dim},
        Δt             :: Real,
        integrator     :: ImplicitIntegrator,
        penalty_method :: Any                    = nothing;
        alg            :: TransferAlgorithm,
        system         :: CoordinateSystem       = DefaultSystem(),
        bc             :: AbstractArray{<: Real} = falses(dim, size(grid)...),
        parallel       :: Bool                   = true,
    ) where {dim}
    consider_boundary_condition = eltype(bc) <: AbstractFloat
    consider_penalty = penalty_method isa PenaltyMethod
    reset_grid_cache!(integrator.grid_cache, integrator.alg, consider_boundary_condition, consider_penalty)
    grid_new = combine(grid, integrator.grid_cache)
    particles_new = combine(particles, integrator.particles_cache)
    solve_grid_velocity!(pt -> (pt.ℂ = update_stress!(pt)), grid_new, particles_new, space, Δt,
                         integrator, penalty_method, alg, system, bc, parallel)
end

function solve_grid_velocity!(
        update_stress! :: Any,
        grid           :: Grid,
        particles      :: Particles,
        space          :: MPSpace,
        Δt             :: Real,
        integrator     :: ImplicitIntegrator,
        penalty_method :: Any,
        alg            :: TransferAlgorithm,
        system         :: CoordinateSystem,
        bc             :: AbstractArray{<: Real},
        parallel       :: Bool,
    )
    α, β, γ = integrator.α, integrator.β, integrator.γ
    freeinds = compute_flatfreeindices(grid, bc)

    # set velocity and acceleration to zero for fixed boundary condition
    impose_fixed_boundary_condition!((grid.vⁿ,grid.aⁿ), bc)

    # calculate `fext` once
    fillzero!(grid.fext)
    particle_to_grid!(:fext, grid, particles, space; alg, system, parallel)

    # friction on boundaries
    consider_boundary_condition = eltype(bc) <: AbstractFloat

    # penalty method
    consider_penalty = penalty_method isa PenaltyMethod

    # jacobian
    should_be_parallel = length(particles) > 200_000 # 200_000 is empirical value
    A = jacobian_matrix(integrator, grid, particles, space, Δt, freeinds, consider_boundary_condition, consider_penalty, alg, system, parallel)

    function residual_jacobian!(R, J, x)
        flatview(grid.u, freeinds) .= x
        @. grid.a = (1/(2α*β*Δt^2))*grid.u - (1/(2α*β*Δt))*grid.vⁿ - (1/2β-1)*grid.aⁿ
        @. grid.v = grid.vⁿ + Δt*((1-γ)*grid.aⁿ + γ*grid.a)

        # internal force
        recompute_grid_internal_force!(update_stress!, grid, particles, space; alg, system, parallel)

        # boundary condition
        if consider_boundary_condition
            compute_boundary_friction!(grid, Δt, bc)
            @. grid.fint -= grid.fᵇ
        end

        # penalty force
        if consider_penalty
            compute_penalty_force!(grid, penalty_method, Δt)
            @. grid.fint -= grid.fᵖ
        end

        # residual
        @. grid.R = 2α*β*Δt * (grid.a + (grid.fint - grid.fext) / grid.m)
        R .= flatview(grid.R, freeinds)
    end

    # `v` = `vⁿ` for initial guess
    @. grid.u = Δt*grid.vⁿ + α*Δt^2*(1-(2β/γ))*grid.aⁿ
    u = copy(flatview(grid.u, freeinds))
    converged = solve!(u, residual_jacobian!, similar(u), A, integrator.nlsolver, integrator.linsolve!)
    converged || @warn "Implicit method not converged"

    # @. grid.xⁿ⁺¹ = grid.x + grid.u

    if consider_penalty && penalty_method.storage !== nothing
        @. penalty_method.storage = grid.fᵖ
    end

    nothing
end

function jacobian_matrix(
        integrator                  :: ImplicitIntegrator,
        grid                        :: Grid,
        particles                   :: Particles,
        space                       :: MPSpace,
        Δt                          :: Real,
        freeinds                    :: Vector{<: CartesianIndex},
        consider_boundary_condition :: Bool,
        consider_penalty            :: Bool,
        alg                         :: TransferAlgorithm,
        system                      :: CoordinateSystem,
        parallel                    :: Bool,
    )
    α, β, γ = integrator.α, integrator.β, integrator.γ
    @inline function update_stress!(pt)
        @inbounds pt.σ = (pt.ℂ ⊡ pt.∇u) / pt.V
    end
    LinearMap(length(freeinds)) do Jδu, δu
        @inbounds begin
            flatview(fillzero!(grid.δu), freeinds) .= δu
            recompute_grid_internal_force!(update_stress!,
                                           @rename(grid, δu=>u),
                                           @rename(particles, δσ=>σ),
                                           space;
                                           alg,
                                           system,
                                           parallel)

            # Jacobian-vector product
            grid_δf = grid.fint
            if consider_boundary_condition
                @. grid_δf -= grid.dfᵇdu ⋅ grid.δu
            end
            if consider_penalty
                @. grid_δf -= grid.dfᵖdu ⋅ grid.δu
            end
            δa = flatview(grid_δf ./= grid.m, freeinds)
            @. Jδu = δu/Δt + 2α*β*Δt * δa
        end
    end
end

function recompute_grid_internal_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace; alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    fillzero!(grid.fint)
    blockwise_parallel_each_particle(space, :dynamic; parallel) do p
        @inbounds begin
            pt = LazyRow(particles, p)
            itp = get_interpolation(space)
            mp = values(space, p)
            grid_to_particle!(alg, system, Val((:∇u,)), pt, grid, itp, mp)
            update_stress!(pt)
            particle_to_grid!(alg, system, Val((:fint,)), grid, pt, itp, mp)
        end
    end
end
