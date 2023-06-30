using IterativeSolvers
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
function sparsity_pattern!(cache::SparseMatrixCSCCache{Tv, Ti, T}, m::Integer, n::Integer) where {Tv, Ti, T}
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

struct NewtonSolver{T, JacCache <: Union{Nothing, JacobianCache}}
    jacobian_free::Bool
    maxiter::Int
    tol::T
    θ::T
    linsolve::Function
    R::Vector{T}
    δv::Vector{T}
    jac_cache::JacCache
end

function NewtonSolver(
        ::Type{T},
        gridsize::Dims{dim};
        jacobian_free::Bool = true,
        maxiter::Int = 50,
        tol::Real = sqrt(eps(T)),
        implicit_parameter::Real = 1,
        linsolve = jacobian_free ? default_jacfree_linsolve : default_jacbased_linsolve) where {T, dim}
    jac_cache = jacobian_free ? nothing : JacobianCache(T, gridsize)
    NewtonSolver(jacobian_free, maxiter, T(tol), T(implicit_parameter), linsolve, T[], T[], jac_cache)
end
NewtonSolver(gridsize::Dims; kwargs...) = NewtonSolver(Float64, gridsize; kwargs...)
# helpers
function default_jacfree_linsolve(x, A, b; kwargs...)
    T = eltype(b)
    gmres!(fillzero!(x), A, b; maxiter=15, initially_zero=true, abstol=T(1e-5), reltol=T(1e-5), kwargs...)
end
function default_jacbased_linsolve(x, A, b)
    x .= A \ b
end

function Base.resize!(solver::NewtonSolver, n::Integer)
    resize!(solver.R, n)
    resize!(solver.δv, n)
    solver
end

# convergence
isless_eps(x::Real, p::Int) = abs(x) < eps(typeof(x))^(1/p)
isconverged(x::Real, solver::NewtonSolver) = abs(x) < solver.tol

function compute_flatfreeindices(grid::Grid{dim}, isfixed::AbstractArray{Bool}) where {dim}
    @assert size(isfixed) == (dim, size(grid)...)
    filter(CartesianIndices(isfixed)) do I
        I′ = _tail(I)
        @inbounds isnonzero(grid, I′) && !iszero(grid.m[I′]) && !isfixed[I]
    end
end

function compute_residual!(R::AbstractVector, grid::Grid, Δt::Real, freeinds::AbstractVector{<: CartesianIndex})
    @. grid.δv = grid.v - grid.vⁿ - Δt * (grid.f/grid.m)
    R .= flatarray(grid.δv, freeinds)
end

function recompute_grid_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace; alg::TransferAlgorithm=FLIP(), system::CoordinateSystem=DefaultSystem(), parallel::Bool=true)
    fillzero!(grid.f)
    parallel_each_particle(space; parallel) do p
        @inbounds begin
            pt = LazyRow(particles, p)
            itp = get_interpolation(space)
            mp = values(space, p)
            grid_to_particle!(alg, system, Val((:∇v,)), pt, grid, itp, mp)
            update_stress!(pt)
            particle_to_grid!(alg, system, Val((:f,)), grid, pt, itp, mp)
        end
    end
end
function recompute_grid_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace, solver::NewtonSolver; alg::TransferAlgorithm=FLIP(), system::CoordinateSystem=DefaultSystem(), parallel::Bool=true)
    @. grid.δv = (1-solver.θ)*grid.vⁿ + solver.θ*grid.v
    recompute_grid_force!(update_stress!, @rename(grid, δv=>v), particles, space; alg, system, parallel)
end

# implicit version of grid_to_particle!
function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::NewtonSolver, isfixed::AbstractArray{Bool}; parallel::Bool) where {names}
    @assert :∇v in names
    grid_to_particle!(update_stress!, :∇v, particles, grid, space, Δt, solver, isfixed; alg, system, parallel)
    rest = tuple(delete!(Set(names), :∇v)...)
    !isempty(rest) && grid_to_particle!(rest, particles, grid, space, Δt; alg, system, parallel)
end

function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{(:∇v,)}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::NewtonSolver, isfixed::AbstractArray{Bool}; parallel::Bool)
    # recompute particle stress and grid force
    recompute_grid_force!(update_stress!, grid, particles, space, solver; alg, system, parallel)

    if solver.maxiter != 0
        freeinds = compute_flatfreeindices(grid, isfixed)
        resize!(solver, length(freeinds))

        if solver.jacobian_free
            A = jacobian_free_matrix(solver.θ, grid, particles, space, Δt, freeinds, alg, system, length(particles)>50_000) # 50_000 is empirical value
        else
            A = sparsity_pattern!(solver.jac_cache, space, freeinds)
        end

        v = flatarray(grid.v, freeinds)
        vⁿ = flatarray(grid.vⁿ, freeinds)
        if !isless_eps(maximum(abs, vⁿ), 1)
            δv = solver.δv
            r⁰ = norm(vⁿ)
            @inbounds for k in 1:solver.maxiter
                # compute residual for Newton's method
                R = compute_residual!(solver.R, grid, Δt, freeinds) # reuse δv
                isconverged(norm(R)/r⁰, solver) && return

                ## solve linear equation A⋅δv = -R
                if solver.jacobian_free
                    solver.linsolve(δv, A, rmul!(R, -1))
                    # P = diagonal_preconditioner!(solver.P, solver.θ, particles, space, Δt, freeinds, parallel)
                    # solver.linsolve(δv, A, rmul!(R, -1); Pr=P)
                else
                    jacobian_based_matrix!(A, solver.jac_cache, solver.θ, grid, particles, space, Δt, freeinds, parallel)
                    solver.linsolve(δv, A, rmul!(R, -1))
                end

                isconverged(norm(δv), solver) && return

                # update grid velocity
                @. v += δv

                # recompute particle stress and grid force
                recompute_grid_force!(update_stress!, grid, particles, space, solver; alg, system, parallel)
            end
            @warn "Newton's method not converged"
        end
    end
end

#################
# Jacobian-free #
#################

function jacobian_free_matrix(θ::Real, grid::Grid, particles::Particles, space::MPSpace, Δt::Real, freeinds::Vector{<: CartesianIndex}, alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    @inline function update_stress!(pt)
        @inbounds begin
            ∇δvₚ = pt.∇v
            pt.σ = (pt.ℂ ⊡ ∇δvₚ) / pt.V
        end
    end
    LinearMap(length(freeinds)) do Jδv, δv
        @inbounds begin
            flatarray(fillzero!(grid.δv), freeinds) .= δv
            recompute_grid_force!(update_stress!, @rename(grid, δv=>v, δf=>f), @rename(particles, δσ=>σ), space; alg, system, parallel)
            δa = flatarray(grid.δf ./= grid.m, freeinds)
            @. Jδv = δv - θ * Δt * δa
        end
    end
end

function diagonal_preconditioner!(P::AbstractVector, θ::Real, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, freeinds::Vector{<: CartesianIndex}, parallel::Bool) where {dim}
    @assert legnth(P) == length(freeinds)
    fill!(P, 1)
    fillzero!(grid.δv)
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

##################
# Jacobian-based #
##################

function sparsity_pattern!(jac_cache::JacobianCache, space::MPSpace{dim, T}, freeinds::AbstractVector{<: CartesianIndex}) where {dim, T}
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
    sparsity_pattern!(spmat_cache, m, n)
end

function add!(A::SparseMatrixCSC, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    # `I` must be sorted
    @boundscheck checkbounds(A, I, J)
    @assert length(I) ≤ size(K, 1) && length(J) ≤ size(K, 2)
    # @assert size(K) == map(length, (I, J))
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

    parallel_each_particle(space; parallel) do p
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
