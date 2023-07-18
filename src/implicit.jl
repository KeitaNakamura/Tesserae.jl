using LinearMaps: LinearMap

_tail(I::CartesianIndex) = CartesianIndex(Base.tail(Tuple(I)))

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

@enum LinearizedVariable lin_velocity lin_force

struct ImplicitSolver{T}
    jacobian_free::Bool
    θ::T
    nlsolver::NonlinearSolver
    linsolver::LinearSolver
    grid_cache::StructArray
    pts_cache::StructVector
    jac_cache::Union{Nothing, JacobianCache{T}}
    lin_var::LinearizedVariable
end

function ImplicitSolver(
        ::Type{T},
        grid::SpGrid{dim},
        particles::Particles;
        jacobian_free::Bool = true,
        implicit_parameter::Real = 1,
        tol::Real = 1e-6,
        abstol::Real = tol,
        reltol::Real = tol,
        maxiter::Int = 10,
        nlsolver = NewtonSolver(T; abstol, reltol, maxiter),
        linsolver = jacobian_free ? GMRESSolver(T; maxiter=15, adaptive=true) : LUSolver(),
        lin_var::LinearizedVariable = lin_velocity,
    ) where {T, dim}
    # grid cache
    Tv = eltype(grid.v)
    Tm = Mat{dim,dim,eltype(Tv),dim*dim}
    spinds = get_spinds(grid)
    grid_cache = StructArray(δv = SpArray{Tv}(spinds),
                             fint = SpArray{Tv}(spinds),
                             fext = SpArray{Tv}(spinds),
                             fᶜ = SpArray{Tv}(spinds),
                             dfᶜdf = SpArray{Tm}(spinds))

    # particles cache
    npts = length(particles)
    Tσ = eltype(particles.σ)
    Tℂ = Tensor{Tuple{@Symmetry{3,3}, 3,3}, eltype(Tσ), 4, 54}
    pts_cache = StructArray(δσ=Array{Tσ}(undef, npts), ℂ=Array{Tℂ}(undef, npts))

    # Jacobian cache
    jac_cache = jacobian_free ? nothing : JacobianCache(T, size(grid))

    ImplicitSolver{T}(jacobian_free, implicit_parameter, nlsolver, linsolver, grid_cache, pts_cache, jac_cache, lin_var)
end
ImplicitSolver(grid::Grid, particles::Particles; kwargs...) = ImplicitSolver(Float64, grid, particles; kwargs...)

function reinit_grid_cache!(spgrid::StructArray, friction::Bool)
    n = countnnz(get_spinds(spgrid.δv))
    resize_nonzeros!(spgrid.δv, n)
    resize_nonzeros!(spgrid.fint, n)
    resize_nonzeros!(spgrid.fext, n)
    if friction
        resize_nonzeros!(spgrid.fᶜ, n)
        resize_nonzeros!(spgrid.dfᶜdf, n)
    end
    spgrid
end

## free indices ##

function compute_flatfreeindices(grid::Grid{dim}, isfixed::AbstractArray{Bool}) where {dim}
    @assert size(isfixed) == (dim, size(grid)...)
    filter(CartesianIndices(isfixed)) do I
        I′ = _tail(I)
        @inbounds isactive(grid, I′) && !isfixed[I]
    end
end

## residual ##

function compute_residual!(R::AbstractVector, grid::Grid, Δt::Real, freeinds::AbstractVector{<: CartesianIndex}, solver::ImplicitSolver)
    if solver.lin_var == lin_velocity
        @. grid.δv = grid.v - grid.vⁿ - Δt * ((grid.fint + grid.fext) / grid.m) # reuse δv
        R .= flatarray(grid.δv, freeinds)
    elseif solver.lin_var == lin_force
        θ = solver.θ
        @. grid.δv = grid.v - grid.vⁿ - Δt * ((θ*grid.fint + grid.fext) / grid.m) # reuse δv
        R .= flatarray(grid.δv, freeinds)
    end
end

function compute_flatfreeindices(grid::Grid{dim}, coefs::AbstractArray{<: AbstractFloat}) where {dim}
    @assert size(coefs) == (dim, size(grid)...)
    filter(CartesianIndices(coefs)) do I
        I′ = _tail(I)
        @inbounds isactive(grid, I′) && coefs[I] ≥ 0 # negative friction coefficient means fixed
    end
end

## grid contact force ##

function compute_grid_contact_force!(friction_force_function, grid::Grid, Δt::Real, coefs::AbstractArray{<: AbstractFloat})
    fillzero!(grid.fᶜ)
    fillzero!(grid.dfᶜdf)
    normals = NormalVectorArray(size(grid))
    @inbounds for I in CartesianIndices(coefs)
        μ = coefs[I]
        I′ = _tail(I)
        if μ > 0 && isactive(grid, I′)
            i = nonzeroindex(grid, I′)
            fₙ = grid.fint[i] ⋅ normals[i]
            if fₙ > 0
                dfᶜdf, fᶜ = gradient(grid.fint[i], :all) do f
                    n = normals[i]
                    v★ = grid.vⁿ[i] + Δt * (grid.fext[i] + f) / grid.m[i]
                    vₜ★ = v★ - (v★⋅n) * n
                    vₜ★_norm = norm(vₜ★)
                    if isapproxzero(vₜ★_norm)
                        zero(vₜ★)
                    else
                        τ = friction_force_function(vₜ★_norm, f⋅n, μ)
                        -τ * (vₜ★/vₜ★_norm)
                    end
                end
                grid.fᶜ[i] += fᶜ
                grid.dfᶜdf[i] += dfᶜdf
            end
        end
    end
end

function default_friction_force_function(; ϵᵥ::Real = sqrt(eps(Float64)))
    function(vₜ::Real, fₙ::Real, μ::Real)
        ξ = vₜ / ϵᵥ
        θ = ξ < 1 ? -ξ^2 + 2ξ : one(ξ)
        θ*μ*fₙ
    end
end

## internal force ##

function recompute_grid_internal_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace; alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    fillzero!(grid.fint)
    parallel_each_particle(space; parallel) do p
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

function recompute_grid_internal_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace, solver::ImplicitSolver; alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    if solver.lin_var == lin_velocity
        θ = solver.θ
        @. grid.δv = (1-θ)*grid.vⁿ + θ*grid.v # reuse δv
        recompute_grid_internal_force!(update_stress!, @rename(grid, δv=>v), particles, space; alg, system, parallel)
    elseif solver.lin_var == lin_force
        recompute_grid_internal_force!(update_stress!, grid, particles, space; alg, system, parallel)
    end
end

## implicit version of grid_to_particle! ##

function grid_to_particle!(
        update_stress!,
        alg       :: TransferAlgorithm,
        system    :: CoordinateSystem,
                  :: Val{names},
        particles :: Particles,
        grid      :: Grid,
        space     :: MPSpace,
        Δt        :: Real,
        solver    :: ImplicitSolver,
        cond      :: AbstractArray{<: Real},
        friction_force_function = default_friction_force_function();
        parallel  :: Bool) where {names}
    @assert :∇v in names
    grid_to_particle!(update_stress!, :∇v, particles, grid, space, Δt, solver, cond, friction_force_function; alg, system, parallel)
    rest = tuple(delete!(Set(names), :∇v)...)
    !isempty(rest) && grid_to_particle!(rest, particles, grid, space, Δt; alg, system, parallel)
end

function grid_to_particle!(
        update_stress!,
        alg       :: TransferAlgorithm,
        system    :: CoordinateSystem,
                  :: Val{(:∇v,)},
        particles :: Particles,
        grid      :: Grid,
        space     :: MPSpace,
        Δt        :: Real,
        solver    :: ImplicitSolver,
        cond      :: AbstractArray{<: Real},
        friction_force_function;
        parallel  :: Bool
    )
    consider_friction = eltype(cond) <: AbstractFloat
    _grid_to_particle!(pt -> (pt.ℂ = update_stress!(pt)),
                       alg,
                       system,
                       Val((:∇v,)),
                       combine(particles, solver.pts_cache),
                       combine(grid, reinit_grid_cache!(solver.grid_cache, consider_friction)),
                       space,
                       Δt,
                       solver,
                       cond,
                       friction_force_function,
                       parallel)
end

function _grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{(:∇v,)}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::ImplicitSolver, cond::AbstractArray{<: Real}, friction_force_function, parallel::Bool)
    @assert :δv in propertynames(grid) && :fint in propertynames(grid) && :fext in propertynames(grid)
    @assert :δσ in propertynames(particles) && :ℂ in propertynames(particles)

    θ = solver.θ
    freeinds = compute_flatfreeindices(grid, cond)

    # calculate fext once
    fillzero!(grid.fext)
    particle_to_grid!(:fext, grid, particles, space; alg, system, parallel)
    if solver.lin_var == lin_force
        @. grid.fext = (1-θ)*grid.f + θ*grid.fext
    end

    # friction on boundaries
    consider_friction = eltype(cond) <: AbstractFloat

    # jacobian
    if solver.jacobian_free
        should_be_parallel = length(particles) > 200_000 # 200_000 is empirical value
        A = jacobian_free_matrix(θ, grid, particles, space, Δt, freeinds, consider_friction, alg, system, parallel)
    else
        A = construct_sparse_matrix!(solver.jac_cache, space, freeinds)
    end

    function residual_jacobian!(R, J, x)
        # internal force
        recompute_grid_internal_force!(update_stress!, grid, particles, space, solver; alg, system, parallel)

        # contact force
        if consider_friction
            compute_grid_contact_force!(friction_force_function, grid, Δt, cond)
            @. grid.fint += grid.fᶜ
        end

        # residual
        compute_residual!(R, grid, Δt, freeinds, solver)

        # jacobian
        if !solver.jacobian_free
            jacobian_based_matrix!(J, solver.jac_cache, θ, grid, particles, space, Δt, freeinds, parallel)
        end
    end

    v = flatarray(grid.v, freeinds)
    converged = solve!(v, residual_jacobian!, similar(v), A, solver.nlsolver, solver.linsolver)
    converged || @warn "Implicit method not converged"
end

## for Jacobian-free method ##

function jacobian_free_matrix(θ::Real, grid::Grid, particles::Particles, space::MPSpace, Δt::Real, freeinds::Vector{<: CartesianIndex}, consider_friction::Bool, alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    @inline function update_stress!(pt)
        @inbounds begin
            ∇δvₚ = pt.∇v
            pt.σ = (pt.ℂ ⊡ ∇δvₚ) / pt.V
        end
    end
    LinearMap(length(freeinds)) do Jδv, δv
        @inbounds begin
            flatarray(fillzero!(grid.δv), freeinds) .= δv
            recompute_grid_internal_force!(update_stress!, @rename(grid, δv=>v), @rename(particles, δσ=>σ), space; alg, system, parallel)
            @. grid.fint *= θ
            if consider_friction
                @. grid.fint += grid.dfᶜdf ⋅ grid.fint
            end
            δa = flatarray(grid.fint ./= grid.m, freeinds)
            @. Jδv = δv - Δt * δa
        end
    end
end

function diagonal_preconditioner!(P::AbstractVector, θ::Real, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, freeinds::Vector{<: CartesianIndex}, parallel::Bool) where {dim}
    @assert legnth(P) == length(freeinds)
    fill!(P, 1)
    fillzero!(grid.δv) # reuse δv
    parallel_each_particle(space; parallel) do p
        @_inline_meta
        @inbounds begin
            ℂₚ = Tensorial.resizedim(particles.ℂ[p], Val(dim))
            mp = values(space, p)
            gridindices = neighbornodes(mp, grid)
            @simd for j in CartesianIndices(gridindices)
                i = gridindices[j]
                ∇N = mp.∇N[j]
                grid.δv[i] += diag(∇N ⋅ ℂₚ ⋅ ∇N)
            end
        end
    end
    @. grid.δv *= θ * Δt / grid.m
    Diagonal(broadcast!(+, P, P, flatarray(grid.δv, freeinds)))
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
    flatarray(griddofs)[freeinds] .= 1:length(freeinds)

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

function jacobian_based_matrix!(K::SparseMatrixCSC, jac_cache::JacobianCache, θ::Real, grid::Grid{dim}, particles::Particles, space::MPSpace{dim, T}, Δt::Real, freeinds::Vector{<: CartesianIndex}, parallel::Bool) where {dim, T}
    @assert size(jac_cache.griddofs) == size(grid) == gridsize(space)
    fillzero!(K)
    griddofs = reinterpret(reshape, Int, jac_cache.griddofs) # flatten dofs assinged on grid
    rng(i) = dim*(i-1)+1 : dim*i

    # thread-local storages
    nₚ = dim * prod(gridsize(get_interpolation(space), Val(dim)))
    Kₚ_threads = [Array{T}(undef, nₚ, nₚ) for _ in 1:Threads.nthreads()]
    dofs_threads = [Int[] for _ in 1:Threads.nthreads()]

    parallel_each_particle_static(space; parallel) do p
        @_inline_meta
        @inbounds begin
            ℂₚ = Tensorial.resizedim(particles.ℂ[p], Val(dim))
            mp = values(space, p)
            gridindices = neighbornodes(mp)
            gridindices_local = CartesianIndices(gridindices)

            # reinit Kₚ
            Kₚ = Kₚ_threads[Threads.threadid()]
            fillzero!(Kₚ)

            # assemble Kₚ
            for (j, l) in enumerate(gridindices_local)
                ∇Nⱼ = mp.∇N[l]
                for (i, k) in enumerate(gridindices_local)
                    ∇Nᵢ = mp.∇N[k]
                    Kₚ[rng(i), rng(j)] .= SArray(∇Nᵢ ⋅ ℂₚ ⋅ ∇Nⱼ)
                end
            end

            # generate local dofs
            dofs = dofs_threads[Threads.threadid()]
            copy!(dofs, vec(view(griddofs, :, gridindices)))
            linds = findall(!iszero, dofs) # dof could be zero when the boundary conditions are imposed

            # add Kₚ to global matrix
            if length(linds) == length(dofs)
                add!(K, dofs, dofs, Kₚ)
            else # for boundaries
                add!(K, view(dofs, linds), view(dofs, linds), view(Kₚ, linds, linds))
            end
        end
    end
    rmul!(K, θ * Δt)

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
                vals[i] += 1 # +1 for diagonal entries
            end
        end
    end

    K
end
