using NewtonSolvers
using IterativeSolvers
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
    fillzero!(grid.dfᵇdu)
    normals = NormalVectorArray(size(grid))
    for I in CartesianIndices(coefs)
        μ = coefs[I]
        I′ = _tail(I)
        if μ > 0 && isactive(grid, I′)
            i = nonzeroindex(grid, I′)
            n = normals[i]
            fint = grid.fint[i]
            if fint ⋅ n < 0
                dfᵇdu, fᵇ = gradient(grid.u[i], :all) do u
                    fₙ = normal(fint,n)
                    uₜ = tangential(u,n)
                    fₙ_norm = norm(fₙ)
                    uₜ_norm = norm(uₜ)
                    (isapproxzero(fₙ_norm) || isapproxzero(uₜ_norm)) && return fₙ
                    f̄ₜ = grid.m[i] * uₜ / Δt^2
                    -min(1, μ*fₙ_norm/norm(f̄ₜ)) * f̄ₜ
                end
                grid.fᵇ[i] += fᵇ
                grid.dfᵇdu[i] += dfᵇdu
            end
        end
    end
end

struct SparseMatrixCSCCache{Tv, Ti, T}
    I::Vector{Ti}
    J::Vector{Ti}
    V::Vector{Tv}
    V′::Vector{T}
    csrrowptr::Vector{Ti}
    csrcolval::Vector{Ti}
    csrnzval::Vector{Tv}
    klasttouch::Vector{Ti}
end
function SparseMatrixCSCCache{Tv, Ti, T}() where {Tv, Ti, T}
    SparseMatrixCSCCache(Ti[], Ti[], Tv[], T[], Ti[], Ti[], Tv[], Ti[])
end
function construct_sparse_matrix!(cache::SparseMatrixCSCCache{Tv, Ti, T}, m::Integer, n::Integer) where {Tv, Ti, T}
    I, J, V = cache.I, cache.J, cache.V
    csrrowptr = cache.csrrowptr
    csrcolval = cache.csrcolval
    csrnzval = cache.csrnzval
    klasttouch = cache.klasttouch
    coolen = length(I)
    resize!(csrrowptr, m+1)
    resize!(csrcolval, coolen)
    resize!(csrnzval, coolen)
    resize!(klasttouch, n)
    S = SparseArrays.sparse!(I, J, V, m, n, |, klasttouch,
                             csrrowptr, csrcolval, csrnzval,
                             I, J, V)
    fillzero!(resize!(cache.V′, length(S.nzval)))
    SparseMatrixCSC{T, Ti}(m, n, S.colptr, S.rowval, cache.V′)
end

# for Jacobian-based method
struct JacobianCache{T, dim, N}
    griddofs::Array{Vec{dim, Int}, dim}
    spmat_cache::SparseMatrixCSCCache{Bool, Int, T}
    spmat_grid_mask::Array{Bool, N}
end
function JacobianCache(::Type{T}, gridsize::Dims{dim}) where {T, dim}
    griddofs = Array{Vec{dim, Int}}(undef, gridsize)
    spmat_cache = SparseMatrixCSCCache{Bool, Int, T}()
    spmat_grid_mask = Array{Bool}(undef, gridsize..., gridsize...)
    JacobianCache(griddofs, spmat_cache, spmat_grid_mask)
end

struct ImplicitIntegrator{T}
    alg::TimeIntegrationAlgorithm
    α::T
    β::T
    γ::T
    nlsolve::Function
    grid_cache::StructArray
    particles_cache::StructVector
    jac_cache::Union{Nothing, JacobianCache{T}}
end

function ImplicitIntegrator(
                      :: Type{T},
        alg           :: TimeIntegrationAlgorithm,
        grid          :: Grid,
        particles     :: Particles;
        jacobian_free :: Bool = true,
        f_tol         :: Real = zero(T),
        x_tol         :: Real = convert(T, 1e-8),
        dx_tol        :: Real = eps(T),
        iterations    :: Int  = 1000,
        linsolve      :: Any  = jacobian_free ? (x,A,b)->idrs!(x,A,b;abstol=x_tol/10,reltol=zero(T)) : (x,A,b)->x.=A\b,
        backtracking  :: Bool = true,
        showtrace     :: Bool = false,
        logall        :: Bool = false,
    ) where {T}
    α, β, γ = integration_parameters(alg)
    nlsolve(f!, j!, f, j, x) = NewtonSolvers.solve!(f!, j!, f, j, x; f_tol, x_tol, dx_tol, iterations, linsolve, backtracking, showtrace, logall)
    grid_cache = create_grid_cache(grid, alg)
    particles_cache = fillzero!(create_particles_cache(particles, alg))
    jac_cache = jacobian_free ? nothing : JacobianCache(T, size(grid))
    ImplicitIntegrator{T}(alg, α, β, γ, nlsolve, grid_cache, particles_cache, jac_cache)
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
        grid           :: Grid{dim},
        particles      :: Particles,
        space          :: MPSpace{dim},
        Δt             :: Real,
        integrator     :: ImplicitIntegrator,
        penalty_method :: Any,
        alg            :: TransferAlgorithm,
        system         :: CoordinateSystem,
        bc             :: AbstractArray{<: Real},
        parallel       :: Bool,
    ) where {dim}
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
    if integrator.jac_cache === nothing
        should_be_parallel = length(particles)*dim > 500_000 # 500_000 is empirical value
        A = jacobian_free_matrix(integrator, grid, particles, space, Δt, freeinds, consider_boundary_condition, consider_penalty, alg, system, parallel)
    else
        A = construct_sparse_matrix!(integrator.jac_cache, space, freeinds)
    end

    function residual!(R, x)
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
        @. grid.R = 2α*β*Δt^2 * (grid.a + (grid.fint - grid.fext) / grid.m)
        R .= flatview(grid.R, freeinds)
    end
    function jacobian!(J, x)
        if integrator.jac_cache !== nothing
            jacobian_based_matrix!(J, integrator, grid, particles, space, Δt, freeinds, consider_boundary_condition, consider_penalty, parallel)
        end
    end

    # `v` = `vⁿ` for initial guess
    @. grid.u = Δt*grid.vⁿ + α*Δt^2*(1-(2β/γ))*grid.aⁿ
    u = copy(flatview(grid.u, freeinds))
    ch = integrator.nlsolve(residual!, jacobian!, similar(u), A, u)
    ch.isconverged || @warn "Implicit method not converged"

    @. grid.x = grid.X + grid.u

    if consider_penalty && penalty_method.storage !== nothing
        @. penalty_method.storage = grid.fᵖ
    end

    ch
end

function jacobian_free_matrix(
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
            @. Jδu = δu + 2α*β*Δt^2 * δa
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

## for Jacobian-based method ##

function construct_sparse_matrix!(jac_cache::JacobianCache, space::MPSpace{dim, T}, freeinds::AbstractVector{<: CartesianIndex}) where {dim, T}
    griddofs = jac_cache.griddofs
    spmat_grid_mask = jac_cache.spmat_grid_mask
    spmat_cache = jac_cache.spmat_cache
    I, J, V = spmat_cache.I, spmat_cache.J, spmat_cache.V
    @assert size(griddofs) == gridsize(space)

    # reinit griddofs
    fillzero!(griddofs)
    flatview(griddofs, freeinds) .= 1:length(freeinds)

    nelts = prod(gridsize(space) .- 1)
    nₚ = dim * prod(gridsize(get_interpolation(space), Val(dim)))
    len = nelts * nₚ^2 # roughly compute enough length

    resize!(I, len)
    resize!(J, len)
    count = 1
    spmat_grid_mask .= false
    gridindices_prev = CartesianIndices((1:0,1:0))
    @inbounds for p in 1:num_particles(space)
        gridindices = neighbornodes(values(space, p))
        if gridindices !== gridindices_prev
            for grid_j in gridindices
                dofs_j = griddofs[grid_j]
                for grid_i in gridindices
                    dofs_i = griddofs[grid_i]
                    if !(spmat_grid_mask[grid_i, grid_j])
                        for jᵈ in 1:dim, iᵈ in 1:dim
                            i = dofs_i[iᵈ]
                            j = dofs_j[jᵈ]
                            I[count] = i
                            J[count] = j
                            count += !iszero(i*j)
                        end
                        spmat_grid_mask[grid_i, grid_j] = true
                    end
                end
            end
            gridindices_prev = gridindices
        end
    end

    resize!(I, count-1)
    resize!(J, count-1)
    resize!(V, count-1)
    V .= true

    m = n = length(freeinds)
    construct_sparse_matrix!(spmat_cache, m, n)
end

function add!(A::SparseMatrixCSC, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    # `I` must be sorted
    @boundscheck checkbounds(A, I, J)
    @assert size(K) == map(length, (I, J))
    rows = rowvals(A)
    vals = nonzeros(A)
    perm = sortperm(I)
    @inbounds for j in 1:length(J)
        i = 1
        for k in nzrange(A, J[j])
            row = rows[k] # row candidate
            i′ = perm[i]
            if I[i′] == row
                vals[k] += K[i′,j]
                i += 1
            end
            i > length(I) && break
        end
        if i <= length(I)
            error("wrong sparsity pattern")
        end
    end
    A
end

function jacobian_based_matrix!(
        K                           :: SparseMatrixCSC,
        integrator                  :: ImplicitIntegrator,
        grid                        :: Grid{dim},
        particles                   :: Particles,
        space                       :: MPSpace{dim, T},
        Δt                          :: Real,
        freeinds                    :: Vector{<: CartesianIndex},
        consider_boundary_condition :: Bool,
        consider_penalty            :: Bool,
        parallel                    :: Bool,
    ) where {dim, T}
    jac_cache = integrator.jac_cache
    @assert size(jac_cache.griddofs) == size(grid) == gridsize(space)
    fillzero!(K)
    griddofs = reinterpret(reshape, Int, jac_cache.griddofs) # flatten dofs assinged on grid
    rng(i) = dim*(i-1)+1 : dim*i

    # thread-local storages
    nₚ = dim * prod(gridsize(get_interpolation(space), Val(dim)))
    Kₚ = Array{T}(undef, nₚ, nₚ)
    dofs = Int[]

    α, β, γ = integrator.α, integrator.β, integrator.γ
    blockwise_parallel_each_particle(space, :static; parallel=false) do p
        @inbounds begin
            ℂₚ = Tensorial.resizedim(particles.ℂ[p], Val(dim))
            mp = values(space, p)
            gridindices = neighbornodes(mp)
            gridindices_local = CartesianIndices(gridindices)

            # assemble Kₚ
            fillzero!(Kₚ)
            for (j, l) in enumerate(gridindices_local)
                ∇Nⱼ = mp.∇N[l]
                for (i, k) in enumerate(gridindices_local)
                    ∇Nᵢ = mp.∇N[k]
                    Kₚ[rng(i), rng(j)] .= SArray(∇Nᵢ ⋅ ℂₚ ⋅ ∇Nⱼ)
                end
            end

            # generate local dofs
            copy!(dofs, vec(view(griddofs, :, gridindices)))
            linds = findall(!iszero, dofs) # dof could be zero when the boundary conditions are imposed

            # add Kₚ to global matrix
            if length(dofs)==nₚ && length(dofs)==length(linds)
                add!(K, dofs, dofs, Kₚ)
            else # for boundaries
                add!(K, view(dofs, linds), view(dofs, linds), view(Kₚ, linds, linds))
            end
        end
    end

    if consider_boundary_condition || consider_penalty
        for I in CartesianIndices(grid)
            if isactive(grid, I)
                i = nonzeroindex(grid, I)
                Kₚ = zero(eltype(grid.dfᵇdu))
                if consider_boundary_condition
                    Kₚ -= grid.dfᵇdu[i]
                end
                if consider_penalty
                    Kₚ -= grid.dfᵖdu[i]
                end

                copy!(dofs, vec(view(griddofs, :, I)))
                linds = findall(!iszero, dofs) # dof could be zero when the boundary conditions are imposed

                if length(dofs) == length(linds)
                    add!(K, dofs, dofs, Kₚ)
                else
                    add!(K, view(dofs, linds), view(dofs, linds), view(Kₚ, linds, linds))
                end
            end
        end
    end

    rmul!(K, 2α*β*Δt^2)

    # compute `K = one(K) + inv(M) * K`
    rows = rowvals(K)
    vals = nonzeros(K)
    @inbounds for j in 1:size(K, 2)
        for i in nzrange(K, j)
            row = rows[i]
            I = freeinds[row]
            mᵢ = grid.m[_tail(I)]
            vals[i] /= mᵢ
            if row == j
                vals[i] += one(eltype(K)) # for diagonal entries
            end
        end
    end

    K
end
