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

struct ImplicitSolver{T}
    jacobian_free::Bool
    θ::T
    nlsolver::NonlinearSolver
    linsolver::LinearSolver
    grid_cache::StructArray
    pts_cache::StructVector
    jac_cache::Union{Nothing, JacobianCache{T}}
end

function ImplicitSolver(
        ::Type{T},
        grid::SpGrid{dim},
        particles::Particles;
        jacobian_free::Bool = true,
        implicit_parameter::Real = 1,
        abstol::Real = 1e-6,
        reltol::Real = 1e-6,
        maxiter::Int = 10,
        nlsolver = NewtonSolver(T; abstol, reltol, maxiter),
        linsolver = jacobian_free ? GMRESSolver(T; maxiter=15, reltol=1e-6, adaptive=true) : LUSolver(),
        friction::Bool = false,
    ) where {T, dim}
    # grid cache
    Tv = eltype(grid.v)
    spinds = get_spinds(grid)
    grid_cache = StructArray(δv   = SpArray{Tv}(spinds),
                             fint = SpArray{Tv}(spinds),
                             fext = SpArray{Tv}(spinds))
    if friction
        Tm = Mat{dim,dim,eltype(grid.v),dim*dim}
        grid_cache = combine(grid_cache, StructArray(dfᶜdv = SpArray{Tm}(spinds),
                                                     dfᶜdf = SpArray{Tm}(spinds)))
    end

    # particles cache
    npts = length(particles)
    Tσ = eltype(particles.σ)
    Tℂ = Tensor{Tuple{@Symmetry{3,3}, 3,3}, eltype(Tσ), 4, 54}
    pts_cache = StructArray(δσ=Array{Tσ}(undef, npts), ℂ=Array{Tℂ}(undef, npts))

    # Jacobian cache
    jac_cache = jacobian_free ? nothing : JacobianCache(T, size(grid))

    ImplicitSolver(jacobian_free, T(implicit_parameter), nlsolver, linsolver, grid_cache, pts_cache, jac_cache)
end
ImplicitSolver(grid::Grid, particles::Particles; kwargs...) = ImplicitSolver(Float64, grid, particles; kwargs...)

function reinit_grid_cache!(spgrid::StructArray)
    n = countnnz(get_spinds(spgrid.δv))
    StructArrays.foreachfield(a->resize_nonzeros!(a,n), spgrid)
    spgrid
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

function compute_residual!(R::AbstractVector, grid::Grid, Δt::Real, freeinds::AbstractVector{<: CartesianIndex})
    @. grid.δv = grid.v - grid.vⁿ - Δt * ((grid.fint + grid.fext) / grid.m) # reuse δv
    R .= flatarray(grid.δv, freeinds)
end

function compute_residual!(R::AbstractVector, grid::Grid, Δt::Real, freeinds::AbstractVector{<: CartesianIndex}, coefs::AbstractArray{<: AbstractFloat})
    # compute frictional forces
    fillzero!(grid.dfᶜdv)
    fillzero!(grid.dfᶜdf)
    @inbounds for I in CartesianIndices(coefs)
        μ = coefs[I]
        if μ > 0 && isactive(grid, I)
            i = nonzeroindex(grid, I)
            (dfᶜdv, dfᶜdf), fᶜ = gradient((v,f) -> compute_friction_force(v, f, grid.m[i], grid.∇m[i], Δt, μ), grid.v[i], grid.fint[i], :all)
            grid.fext[i] += fᶜ
            grid.dfᶜdv[i] = dfᶜdv
            grid.dfᶜdf[i] = dfᶜdf
        end
    end
    compute_residual!(R, grid, Δt, freeinds)
end
function compute_friction_force(v::Vec, fint::Vec, m::Real, n::Vec, Δt::Real, μ::Real)
    f_nor_norm = fint ⋅ n # traction
    if f_nor_norm < 0 # traction is compressive
        f_nor = f_nor_norm * n
        f_stick = -m * v / Δt
        f_stick_norm = norm(f_stick)
        return -f_nor + f_stick * min(μ*f_nor_norm/f_stick_norm, 1)
    else
        return zero(fint)
    end
end

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
function recompute_grid_internal_force!(update_stress!, grid::Grid, particles::Particles, space::MPSpace, θ::Real; alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
    @. grid.δv = (1-θ)*grid.vⁿ + θ*grid.v # reuse δv
    recompute_grid_internal_force!(update_stress!, @rename(grid, δv=>v), particles, space; alg, system, parallel)
end

# implicit version of grid_to_particle!
function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::ImplicitSolver, cond::AbstractArray{<: Real}; parallel::Bool) where {names}
    @assert :∇v in names
    grid_to_particle!(update_stress!, :∇v, particles, grid, space, Δt, solver, cond; alg, system, parallel)
    rest = tuple(delete!(Set(names), :∇v)...)
    !isempty(rest) && grid_to_particle!(rest, particles, grid, space, Δt; alg, system, parallel)
end

function grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{(:∇v,)}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::ImplicitSolver, cond::AbstractArray{<: Real}; parallel::Bool)
    _grid_to_particle!(pt -> (pt.ℂ = update_stress!(pt)),
                       alg,
                       system,
                       Val((:∇v,)),
                       combine(particles, solver.pts_cache),
                       combine(grid, reinit_grid_cache!(solver.grid_cache)),
                       space,
                       Δt,
                       solver,
                       cond,
                       parallel)
end

function _grid_to_particle!(update_stress!, alg::TransferAlgorithm, system::CoordinateSystem, ::Val{(:∇v,)}, particles::Particles, grid::Grid, space::MPSpace, Δt::Real, solver::ImplicitSolver, cond::AbstractArray{<: Real}, parallel::Bool)
    @assert :δv in propertynames(grid) && :fint in propertynames(grid) && :fext in propertynames(grid)
    @assert :δσ in propertynames(particles) && :ℂ in propertynames(particles)

    freeinds = compute_flatfreeindices(grid, cond)

    # calculate fext once
    fillzero!(grid.fext)
    particle_to_grid!(:fext, grid, particles, space; alg, system, parallel)

    # friction on boundaries
    consider_friction = eltype(cond) <: AbstractFloat

    if solver.jacobian_free
        should_be_parallel = length(particles) > 50_000 # 50_000 is empirical value
        A = jacobian_free_matrix(solver.θ, grid, particles, space, Δt, freeinds, consider_friction, alg, system, should_be_parallel)
    else
        A = construct_sparse_matrix!(solver.jac_cache, space, freeinds)
    end

    function residual_jacobian!(R, J, x)
        recompute_grid_internal_force!(update_stress!, grid, particles, space, solver.θ; alg, system, parallel)
        if consider_friction
            compute_residual!(R, grid, Δt, freeinds, cond)
        else
            compute_residual!(R, grid, Δt, freeinds)
        end
        if !solver.jacobian_free
            jacobian_based_matrix!(J, solver.jac_cache, solver.θ, grid, particles, space, Δt, freeinds, parallel)
        end
    end

    v = flatarray(grid.v, freeinds)
    converged = solve!(v, residual_jacobian!, similar(v), A, solver.nlsolver, solver.linsolver)
    converged || @warn "Implicit method not converged"
end

#################
# Jacobian-free #
#################

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
            if consider_friction
                @. grid.fint += grid.dfᶜdv ⋅ grid.δv + grid.dfᶜdf ⋅ grid.fint
            end
            δa = flatarray(grid.fint ./= grid.m, freeinds)
            @. Jδv = δv - θ * Δt * δa
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

##################
# Jacobian-based #
##################

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
