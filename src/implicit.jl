using LinearMaps: LinearMap

_tail(I::CartesianIndex) = CartesianIndex(Base.tail(Tuple(I)))

normal(v::Vec, n::Vec) = (v⋅n) * n
tangential(v::Vec, n::Vec) = v - normal(v,n)

struct EulerIntegrator{T}
    θ::T
    nlsolver::NonlinearSolver
    linsolve!::Function
    grid_cache::StructArray
    pts_cache::StructVector
end

function EulerIntegrator(
        ::Type{T},
        grid::SpGrid{dim},
        particles::Particles;
        implicit_parameter::Real = 1,
        abstol::Real = sqrt(eps(T)),
        reltol::Real = zero(T),
        maxiter::Int = 100,
        linsolve! = (x,A,b) -> gmres!(x,A,b),
    ) where {T, dim}
    # grid cache
    Tv = eltype(grid.v)
    Tm = Mat{dim,dim,eltype(Tv),dim*dim}
    spinds = get_spinds(grid)
    grid_cache = StructArray(v★    = SpArray{Tv}(spinds),
                             f★    = SpArray{Tv}(spinds),
                             fint  = SpArray{Tv}(spinds),
                             fext  = SpArray{Tv}(spinds),
                             fᵇ    = SpArray{Tv}(spinds),
                             dfᵇdf = SpArray{Tm}(spinds),
                             fᵖ    = SpArray{Tv}(spinds),
                             dfᵖdv = SpArray{Tm}(spinds),
                             dfᵖdf = SpArray{Tm}(spinds))

    # particles cache
    npts = length(particles)
    Tσ = eltype(particles.σ)
    Tℂ = Tensor{Tuple{@Symmetry{3,3}, 3,3}, eltype(Tσ), 4, 54}
    pts_cache = StructArray(δσ=Array{Tσ}(undef, npts), ℂ=Array{Tℂ}(undef, npts))

    nlsolver = NewtonSolver(T; abstol, reltol, maxiter)
    EulerIntegrator{T}(implicit_parameter, nlsolver, linsolve!, grid_cache, pts_cache)
end
EulerIntegrator(grid::Grid, particles::Particles; kwargs...) = EulerIntegrator(Float64, grid, particles; kwargs...)

function reinit_grid_cache!(spgrid::StructArray, consider_boundary_condition::Bool, consider_penalty::Bool)
    n = countnnz(get_spinds(spgrid.v★))
    resize_nonzeros!(spgrid.v★, n)
    resize_nonzeros!(spgrid.f★, n)
    resize_nonzeros!(spgrid.fint, n)
    resize_nonzeros!(spgrid.fext, n)
    if consider_boundary_condition
        resize_nonzeros!(spgrid.fᵇ, n)
        resize_nonzeros!(spgrid.dfᵇdf, n)
    end
    if consider_penalty
        resize_nonzeros!(spgrid.fᵖ, n)
        resize_nonzeros!(spgrid.dfᵖdv, n)
        resize_nonzeros!(spgrid.dfᵖdf, n)
    end
    spgrid
end

## fixed boundary condition ##

_isfixed_bc(x::Bool) = x
_isfixed_bc(x::AbstractFloat) = x < 0
function impose_fixed_boundary_condition!(grid::Grid{dim}, bc::AbstractArray{<: Real}) where {dim}
    @assert size(bc) == (dim, size(grid)...)
    for I in CartesianIndices(bc)
        if _isfixed_bc(bc[I])
            flatarray(grid.v)[I] = 0
            flatarray(grid.vⁿ)[I] = 0
        end
    end
end

## free indices ##

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

## boundary friction force ##

function compute_boundary_friction!(grid::Grid, Δt::Real, coefs::AbstractArray{<: AbstractFloat})
    fillzero!(grid.fᵇ)
    fillzero!(grid.dfᵇdf)
    normals = NormalVectorArray(size(grid))
    for I in CartesianIndices(coefs)
        μ = coefs[I]
        I′ = _tail(I)
        if μ > 0 && isactive(grid, I′)
            i = nonzeroindex(grid, I′)
            n = normals[i]
            fint = grid.fint[i]
            if fint ⋅ n > 0
                f̄ₜ = tangential(grid.m[i]*grid.vⁿ[i]/Δt + grid.fext[i], n)
                dfᵇdf, fᵇ = gradient(fint, :all) do f
                    fₙ = normal(f, n)
                    fₜ = f - fₙ
                    fₜ★ = fₜ + f̄ₜ
                    -min(1, μ*norm(fₙ)/norm(fₜ★)) * fₜ★
                end
                grid.fᵇ[i] += fᵇ
                grid.dfᵇdf[i] += dfᵇdf
            end
        end
    end
end

## penalty force ##

struct PenaltyMethod{F <: Function, dim, Grid_g <: AbstractArray{<: Vec{dim}, dim}, Grid_μ <: AbstractArray{<: Real, dim}, Grid_v <: AbstractArray{<: Vec{dim}, dim}, S}
    penalty_force::F
    grid_g::Grid_g
    grid_μ::Grid_μ
    grid_v::Grid_v
    storage::S
end

function PenaltyMethod(
        penalty_force::Function,
        grid_g::AbstractArray{<: Vec},
        grid_μ::AbstractArray{<: Real},
        grid_v::AbstractArray{<: Vec} = FillArray(zero(eltype(grid_g)), size(grid_g));
        storage = nothing,
    )
    PenaltyMethod(penalty_force, grid_g, grid_μ, grid_v, storage)
end

function compute_penalty_force!(grid::Grid, Δt::Real, p::PenaltyMethod)
    compute_penalty_force!(grid, Δt, p.penalty_force, p.grid_g, p.grid_μ, p.grid_v)
end

function compute_penalty_force!(grid::Grid, Δt::Real, penalty_force::Function, grid_g::AbstractArray{<: Vec}, grid_μ::AbstractArray{<: Real}, grid_v::AbstractArray{<: Vec})
    @assert size(grid) == size(grid_g) == size(grid_μ) == size(grid_v)
    fillzero!(grid.fᵖ)
    fillzero!(grid.dfᵖdv)
    fillzero!(grid.dfᵖdf)
    for I in CartesianIndices(grid)
        if isactive(grid, I)
            i = nonzeroindex(grid, I)
            g⁰ = grid_g[i]
            if !iszero(norm(g⁰))
                n = normalize(g⁰)
                μ = grid_μ[i]
                v_rigid = grid_v[i]
                f̄ₜ = tangential(grid.m[i]*(grid.vⁿ[i]-v_rigid)/Δt + grid.fext[i], n)
                (dfᵖdv,dfᵖdf), fᵖ = gradient(grid.v[i], grid.fint[i], :all) do v, f
                    # normal
                    g = g⁰ + normal(v-v_rigid, n) * Δt
                    fₙ = penalty_force(g)

                    # tangential
                    iszero(μ) && return fₙ
                    fₜ★ = tangential(f,n) + f̄ₜ
                    fₜ = -min(1, μ*norm(fₙ)/norm(fₜ★)) * fₜ★

                    fₙ + fₜ
                end
                grid.fᵖ[i] += fᵖ
                grid.dfᵖdv[i] += dfᵖdv
                grid.dfᵖdf[i] += dfᵖdf
            end
        end
    end
end

## internal force ##

function recompute_grid_internal_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace; alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    fillzero!(grid.fint)
    blockwise_parallel_each_particle(space, :dynamic; parallel) do p
        @inbounds begin
            pt = LazyRow(particles, p)
            itp = get_interpolation(space)
            mp = values(space, p)
            grid_to_particle!(alg, system, Val((:∇v,)), pt, grid, itp, mp)
            update_stress!(pt)
            particle_to_grid!(alg, system, Val((:fint,)), grid, pt, itp, mp)
        end
    end
end

## implicit version of grid_to_particle! ##

function default_boundary_condition(gridsize::Dims{dim}) where {dim}
    # normals = NormalVectorArray(gridsize)
    # broadcast(!iszero, reinterpret(reshape, eltype(eltype(normals)), normals))
    falses(dim, gridsize...)
end

function grid_to_particle!(
        update_stress!   :: Any,
        alg              :: TransferAlgorithm,
        system           :: CoordinateSystem,
                         :: Val{names},
        particles        :: Particles,
        grid             :: Grid,
        space            :: MPSpace,
        Δt               :: Real,
        integrator       :: EulerIntegrator,
        penalty_method   :: Union{PenaltyMethod, Nothing} = nothing;
        bc               :: AbstractArray{<: Real}        = default_boundary_condition(size(grid)),
        parallel         :: Bool,
    ) where {names}
    # implicit method
    # up-to-date velocity `grid.v` and interpolated velocity `grid.v★` are calculated
    @assert :∇v in names
    grid_to_particle!(update_stress!, alg, system, Val((:∇v,)), particles, grid, space, Δt, integrator, penalty_method; bc, parallel)

    # use interpolated velocity `grid.v★` to update particle position `x`
    :x in names && grid_to_particle!(:x, particles, @rename(grid, v★=>v), space, Δt; alg, system, parallel)

    # other `v` and `∇v` should be updated by using up-to-date velocity `grid.v`
    rest = tuple(delete!(Set(names), :x)...)
    grid_to_particle!(rest, particles, grid, space, Δt; alg, system, parallel)
end

function grid_to_particle!(
        update_stress! :: Any,
        alg            :: TransferAlgorithm,
        system         :: CoordinateSystem,
                       :: Val{(:∇v,)},
        particles      :: Particles,
        grid           :: Grid,
        space          :: MPSpace,
        Δt             :: Real,
        integrator     :: EulerIntegrator,
        penalty_method :: Union{PenaltyMethod, Nothing} = nothing;
        bc             :: AbstractArray{<: Real}        = default_boundary_condition(size(grid)),
        parallel       :: Bool,
    )
    consider_boundary_condition = eltype(bc) <: AbstractFloat
    consider_penalty = penalty_method isa PenaltyMethod
    _grid_to_particle!(pt -> (pt.ℂ = update_stress!(pt)),
                       alg,
                       system,
                       Val((:∇v,)),
                       combine(particles, integrator.pts_cache),
                       combine(grid, reinit_grid_cache!(integrator.grid_cache, consider_boundary_condition, consider_penalty)),
                       space,
                       Δt,
                       integrator,
                       penalty_method,
                       bc,
                       parallel)
end

function _grid_to_particle!(
        update_stress!   :: Any,
        alg              :: TransferAlgorithm,
        system           :: CoordinateSystem,
                         :: Val{(:∇v,)},
        particles        :: Particles,
        grid             :: Grid,
        space            :: MPSpace,
        Δt               :: Real,
        integrator       :: EulerIntegrator,
        penalty_method   :: Union{PenaltyMethod, Nothing},
        bc               :: AbstractArray{<: Real},
        parallel         :: Bool,
    )
    θ = integrator.θ
    freeinds = compute_flatfreeindices(grid, bc)

    # set velocity to zero for fixed boundary condition
    impose_fixed_boundary_condition!(grid, bc)

    # calculate fext once
    fillzero!(grid.fext)
    particle_to_grid!(:fext, grid, particles, space; alg, system, parallel)

    # friction on boundaries
    consider_boundary_condition = eltype(bc) <: AbstractFloat

    # penalty method
    consider_penalty = penalty_method isa PenaltyMethod

    # jacobian
    should_be_parallel = length(particles) > 200_000 # 200_000 is empirical value
    A = jacobian_matrix(θ, @rename(grid, v★=>δv), particles, space, Δt, freeinds, consider_boundary_condition, consider_penalty, alg, system, parallel)

    function residual_jacobian!(R, J, x)
        flatview(grid.v, freeinds) .= x
        @. grid.v★ = (1-θ)*grid.vⁿ + θ*grid.v

        # internal force
        recompute_grid_internal_force!(update_stress!, @rename(grid, v★=>v), particles, space; alg, system, parallel)

        # boundary condition
        if consider_boundary_condition
            compute_boundary_friction!(grid, Δt, bc)
            @. grid.fint += grid.fᵇ
        end

        # penalty force
        if consider_penalty
            compute_penalty_force!(@rename(grid, v★=>v), Δt, penalty_method)
            @. grid.fint += grid.fᵖ
        end

        # residual
        @. grid.v★ = grid.v - grid.vⁿ - Δt * ((grid.fint + grid.fext) / grid.m) # reuse v★
        R .= flatview(grid.v★, freeinds)
    end

    v = copy(flatview(grid.v, freeinds))
    converged = solve!(v, residual_jacobian!, similar(v), A, integrator.nlsolver, integrator.linsolve!)
    converged || @warn "Implicit method not converged"

    if consider_penalty && penalty_method.storage !== nothing
        @. penalty_method.storage = grid.fᵖ
    end

    # calculate interpolated grid velocity to update `xₚ` outside function
    @. grid.v★ = (1-θ)*grid.vⁿ + θ*grid.v

    nothing
end

# jacobian-free method
function jacobian_matrix(
        θ                           :: Real,
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
    @inline function update_stress!(pt)
        @inbounds begin
            ∇δvₚ = pt.∇v
            pt.σ = (pt.ℂ ⊡ ∇δvₚ) / pt.V
        end
    end
    LinearMap(length(freeinds)) do Jδv, δv
        @inbounds begin
            # setup grid.δv
            flatview(fillzero!(grid.δv), freeinds) .= θ .* δv

            # recompute grid internal force `grid.fint` from grid velocity `grid.v`
            recompute_grid_internal_force!(update_stress!, @rename(grid, δv=>v), @rename(particles, δσ=>σ), space; alg, system, parallel)

            # Jacobian-vector product
            @. grid.f★ = grid.fint
            if consider_boundary_condition
                @. grid.f★ += grid.dfᵇdf ⋅ grid.fint
            end
            if consider_penalty
                @. grid.f★ += grid.dfᵖdv ⋅ grid.δv + grid.dfᵖdf ⋅ grid.fint
            end
            δa = flatview(grid.f★ ./= grid.m, freeinds)
            @. Jδv = δv - Δt * δa
        end
    end
end

function diagonal_preconditioner!(P::AbstractVector, θ::Real, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, freeinds::Vector{<: CartesianIndex}, parallel::Bool) where {dim}
    @assert legnth(P) == length(freeinds)
    fill!(P, 1)
    fillzero!(grid.v★) # reuse v★
    blockwise_parallel_each_particle(space, :dynamic; parallel) do p
        @inbounds begin
            ℂₚ = Tensorial.resizedim(particles.ℂ[p], Val(dim))
            mp = values(space, p)
            gridindices = neighbornodes(mp, grid)
            @simd for j in CartesianIndices(gridindices)
                i = gridindices[j]
                ∇N = mp.∇N[j]
                grid.v★[i] += diag(∇N ⋅ ℂₚ ⋅ ∇N)
            end
        end
    end
    @. grid.v★ *= θ * Δt / grid.m
    Diagonal(broadcast!(+, P, P, flatview(grid.v★, freeinds)))
end
