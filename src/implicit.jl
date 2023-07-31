using LinearMaps: LinearMap

_tail(I::CartesianIndex) = CartesianIndex(Base.tail(Tuple(I)))

normal(v::Vec, n::Vec) = (v⋅n) * n
tangential(v::Vec, n::Vec) = v - normal(v,n)

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
        tol::Real = 1e-6,
        abstol::Real = tol,
        reltol::Real = tol,
        maxiter::Int = 10,
        nlsolver = NewtonSolver(T; abstol, reltol, maxiter),
        linsolver = jacobian_free ? GMRESSolver(T; maxiter=15, adaptive=true) : LUSolver(),
    ) where {T, dim}
    # grid cache
    Tv = eltype(grid.v)
    Tm = Mat{dim,dim,eltype(Tv),dim*dim}
    spinds = get_spinds(grid)
    grid_cache = StructArray(δv    = SpArray{Tv}(spinds),
                             f★    = SpArray{Tv}(spinds),
                             fint  = SpArray{Tv}(spinds),
                             fext  = SpArray{Tv}(spinds),
                             fᵇ    = SpArray{Tv}(spinds),
                             dfᵇdf = SpArray{Tm}(spinds))

    # particles cache
    npts = length(particles)
    Tσ = eltype(particles.σ)
    Tℂ = Tensor{Tuple{@Symmetry{3,3}, 3,3}, eltype(Tσ), 4, 54}
    pts_cache = StructArray(δσ=Array{Tσ}(undef, npts), ℂ=Array{Tℂ}(undef, npts))

    # Jacobian cache
    jac_cache = jacobian_free ? nothing : JacobianCache(T, size(grid))

    ImplicitSolver{T}(jacobian_free, implicit_parameter, nlsolver, linsolver, grid_cache, pts_cache, jac_cache)
end
ImplicitSolver(grid::Grid, particles::Particles; kwargs...) = ImplicitSolver(Float64, grid, particles; kwargs...)

function reinit_grid_cache!(spgrid::StructArray, consider_boundary_condition::Bool)
    n = countnnz(get_spinds(spgrid.δv))
    resize_nonzeros!(spgrid.δv, n)
    resize_nonzeros!(spgrid.f★, n)
    resize_nonzeros!(spgrid.fint, n)
    resize_nonzeros!(spgrid.fext, n)
    if consider_boundary_condition
        resize_nonzeros!(spgrid.fᵇ, n)
        resize_nonzeros!(spgrid.dfᵇdf, n)
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
        update_stress!,
        alg             :: TransferAlgorithm,
        system          :: CoordinateSystem,
                        :: Val{names},
        particles       :: Particles,
        grid            :: Grid,
        space           :: MPSpace,
        Δt              :: Real,
        solver          :: ImplicitSolver;
        bc              :: AbstractArray{<: Real} = default_boundary_condition(size(grid)),
        parallel        :: Bool,
    ) where {names}
    @assert :∇v in names
    grid_to_particle!(update_stress!, alg, system, Val((:∇v,)), particles, grid, space, Δt, solver; bc, parallel)
    rest = tuple(delete!(Set(names), :∇v)...)
    !isempty(rest) && grid_to_particle!(rest, particles, grid, space, Δt; alg, system, parallel)
end

function grid_to_particle!(
        update_stress!,
        alg             :: TransferAlgorithm,
        system          :: CoordinateSystem,
                        :: Val{(:∇v,)},
        particles       :: Particles,
        grid            :: Grid,
        space           :: MPSpace,
        Δt              :: Real,
        solver          :: ImplicitSolver;
        bc              :: AbstractArray{<: Real} = default_boundary_condition(size(grid)),
        parallel        :: Bool,
    )
    consider_boundary_condition = eltype(bc) <: AbstractFloat
    _grid_to_particle!(pt -> (pt.ℂ = update_stress!(pt)),
                       alg,
                       system,
                       Val((:∇v,)),
                       combine(particles, solver.pts_cache),
                       combine(grid, reinit_grid_cache!(solver.grid_cache, consider_boundary_condition)),
                       space,
                       Δt,
                       solver,
                       bc,
                       parallel)
end

function _grid_to_particle!(
        update_stress!,
        alg             :: TransferAlgorithm,
        system          :: CoordinateSystem,
                        :: Val{(:∇v,)},
        particles       :: Particles,
        grid            :: Grid,
        space           :: MPSpace,
        Δt              :: Real,
        solver          :: ImplicitSolver,
        bc              :: AbstractArray{<: Real},
        parallel        :: Bool,
    )
    @assert :δv in propertynames(grid) && :fint in propertynames(grid) && :fext in propertynames(grid)
    @assert :δσ in propertynames(particles) && :ℂ in propertynames(particles)

    θ = solver.θ
    freeinds = compute_flatfreeindices(grid, bc)

    # set velocity to zero for fixed boundary condition
    impose_fixed_boundary_condition!(grid, bc)

    # calculate fext once
    fillzero!(grid.fext)
    particle_to_grid!(:fext, grid, particles, space; alg, system, parallel)

    # friction on boundaries
    consider_boundary_condition = eltype(bc) <: AbstractFloat

    # jacobian
    if solver.jacobian_free
        should_be_parallel = length(particles) > 200_000 # 200_000 is empirical value
        A = jacobian_free_matrix(θ, grid, particles, space, Δt, freeinds, consider_boundary_condition, alg, system, parallel)
    else
        A = construct_sparse_matrix!(solver.jac_cache, space, freeinds)
    end

    function residual_jacobian!(R, J, x)
        flatview(grid.v, freeinds) .= x
        @. grid.δv = (1-θ)*grid.vⁿ + θ*grid.v # reuse δv

        # internal force
        recompute_grid_internal_force!(update_stress!, @rename(grid, δv=>v), particles, space; alg, system, parallel)

        # boundary condition
        if consider_boundary_condition
            compute_boundary_friction!(grid, Δt, bc)
            @. grid.fint += grid.fᵇ
        end

        # residual
        @. grid.δv = grid.v - grid.vⁿ - Δt * ((grid.fint + grid.fext) / grid.m) # reuse δv
        R .= flatview(grid.δv, freeinds)

        # jacobian
        if !solver.jacobian_free
            jacobian_based_matrix!(J, solver.jac_cache, θ, grid, particles, space, Δt, freeinds, parallel)
        end
    end

    v = copy(flatview(grid.v, freeinds))
    converged = solve!(v, residual_jacobian!, similar(v), A, solver.nlsolver, solver.linsolver)
    converged || @warn "Implicit method not converged"
end

## for Jacobian-free method ##

function jacobian_free_matrix(θ::Real, grid::Grid, particles::Particles, space::MPSpace, Δt::Real, freeinds::Vector{<: CartesianIndex}, consider_boundary_condition::Bool, alg::TransferAlgorithm, system::CoordinateSystem, parallel::Bool)
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
            δa = flatview(grid.f★ ./= grid.m, freeinds)
            @. Jδv = δv - Δt * δa
        end
    end
end

function diagonal_preconditioner!(P::AbstractVector, θ::Real, particles::Particles, grid::Grid{dim}, space::MPSpace{dim}, Δt::Real, freeinds::Vector{<: CartesianIndex}, parallel::Bool) where {dim}
    @assert legnth(P) == length(freeinds)
    fill!(P, 1)
    fillzero!(grid.δv) # reuse δv
    blockwise_parallel_each_particle(space, :dynamic; parallel) do p
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
    Diagonal(broadcast!(+, P, P, flatview(grid.δv, freeinds)))
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

function jacobian_based_matrix!(K::SparseMatrixCSC, jac_cache::JacobianCache, θ::Real, grid::Grid{dim}, particles::Particles, space::MPSpace{dim, T}, Δt::Real, freeinds::Vector{<: CartesianIndex}, parallel::Bool) where {dim, T}
    @assert size(jac_cache.griddofs) == size(grid) == gridsize(space)
    fillzero!(K)
    griddofs = reinterpret(reshape, Int, jac_cache.griddofs) # flatten dofs assinged on grid
    rng(i) = dim*(i-1)+1 : dim*i

    # thread-local storages
    nₚ = dim * prod(gridsize(get_interpolation(space), Val(dim)))
    Kₚ_threads = [Array{T}(undef, nₚ, nₚ) for _ in 1:Threads.nthreads()]
    dofs_threads = [Int[] for _ in 1:Threads.nthreads()]

    blockwise_parallel_each_particle(space, :static; parallel) do p
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
