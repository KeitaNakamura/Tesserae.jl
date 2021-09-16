struct MPCache{dim, T, Tshape <: ShapeValues{dim, T}}
    shapevalues::Vector{Tshape}
    gridsize::NTuple{dim, Int}
    gridindices::Vector{Vector{Index{dim}}}
    pointsinblock::Array{Vector{Int}, dim}
    nearsurface::BitVector
end

function MPCache(grid::Grid{dim, T}, xₚ::AbstractVector) where {dim, T}
    checkshapefunction(grid)
    npoints = length(xₚ)
    shapevalues = [ShapeValues(T, grid.shapefunction) for _ in 1:npoints]
    gridindices = [Index{dim}[] for _ in 1:npoints]
    MPCache(shapevalues, size(grid), gridindices, pointsinblock(grid, xₚ), falses(npoints))
end

npoints(cache::MPCache) = length(cache.nearsurface)
gridsize(cache::MPCache) = cache.gridsize

function reordering_pointstate!(pointstate::AbstractVector, cache::MPCache)
    inds = Vector{Int}(undef, length(pointstate))
    cnt = 1
    for block in cache.pointsinblock
        @inbounds for i in eachindex(block)
            inds[cnt] = block[i]
            block[i] = cnt
            cnt += 1
        end
    end
    @. pointstate = pointstate[inds]
    pointstate
end

function allocate!(f, x::Vector, n::Integer)
    len = length(x)
    if n > len # growend
        resize!(x, n)
        @simd for i in len+1:n
            @inbounds x[i] = f(i)
        end
    end
    x
end

function update!(cache::MPCache{dim}, grid::Grid{dim}, xₚ::AbstractVector; exclude = nothing) where {dim}
    checkshapefunction(grid)
    @assert size(grid) == gridsize(cache)

    allocate!(i -> eltype(cache.shapevalues)(), cache.shapevalues, length(xₚ))
    allocate!(i -> Index{dim}[], cache.gridindices, length(xₚ))
    resize!(cache.nearsurface, length(xₚ))

    gridstate = grid.state
    pointsinblock!(cache.pointsinblock, grid, xₚ)

    spat = sparsity_pattern(grid, xₚ, cache.pointsinblock; exclude)
    cache.nearsurface .= false
    @inbounds Threads.@threads for p in eachindex(xₚ)
        x = xₚ[p]
        gridindices = cache.gridindices[p]
        cache.nearsurface[p] = neighboring_nodes!(gridindices, grid, x, spat)
        update!(cache.shapevalues[p], grid, x, gridindices)
    end

    gridstate.spat .= spat
    reinit!(gridstate)

    cache
end

##################
# point_to_grid! #
##################

@generated function _point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray, N}}, shapevalues::ShapeValues, gridindices::Vector{<: Index}, p::Int) where {N}
    exps = [:(gridstates[$i][I] += res[$i]) for i in 1:N]
    quote
        @inbounds @simd for i in eachindex(shapevalues, gridindices)
            it = shapevalues[i]
            I = gridindices[i]
            res = p2g(it, p, I)
            $(exps...)
        end
    end
end

function point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray}}, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    @assert all(==(gridsize(cache)), size.(gridstates))
    pointmask !== nothing && @assert length(pointmask) == npoints(cache)
    for color in coloringblocks(gridsize(cache))
        Threads.@threads for blockindex in color
            @inbounds for p in cache.pointsinblock[blockindex]
                pointmask !== nothing && !pointmask[p] && continue
                _point_to_grid!(p2g, gridstates, cache.shapevalues[p], cache.gridindices[p], p)
            end
        end
    end
    gridstates
end

function point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray}}, grid::Grid{dim, T}, xₚ::AbstractVector; exclude = nothing) where {dim, T}
    @assert all(==(size(grid)), size.(gridstates))
    ptsinblk = pointsinblock(grid, xₚ)
    spat = sparsity_pattern(grid, xₚ, ptsinblk; exclude)
    shapevalues_threads = [ShapeValues(T, grid.shapefunction) for _ in 1:Threads.nthreads()]
    gridindices_threads = [Index{dim}[] for _ in 1:Threads.nthreads()]
    for color in coloringblocks(size(grid))
        Threads.@threads for blockindex in color
            shapevalues = shapevalues_threads[Threads.threadid()]
            gridindices = gridindices_threads[Threads.threadid()]
            for p in ptsinblk[blockindex]
                x = xₚ[p]
                neighboring_nodes!(gridindices, grid, x, spat)
                update!(shapevalues, grid, x, gridindices)
                _point_to_grid!(p2g, gridstates, shapevalues, gridindices, p)
            end
        end
    end
    gridstates
end

function point_to_grid!(p2g, gridstate::AbstractArray, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    point_to_grid!((gridstate,), cache, pointmask) do it, p, I
        @_inline_meta
        @_propagate_inbounds_meta
        (p2g(it, p, I),)
    end
end

function point_to_grid!(p2g, gridstate::AbstractArray, grid::Grid, xₚ::AbstractVector; exclude = nothing)
    point_to_grid!((gridstate,), grid, xₚ; exclude) do it, p, I
        @_inline_meta
        @_propagate_inbounds_meta
        (p2g(it, p, I),)
    end
end

function stress_to_force(coord_system::Symbol, N, ∇N, x::Vec, σ::SymmetricSecondOrderTensor{3})
    f = Tensor2D(σ) ⋅ ∇N
    @inbounds coord_system == :axisymmetric ? f + Vec(1,0)*σ[3,3]*N/x[1] : f
end

function default_point_to_grid!(grid::Grid{<: Any, <: Any, <: WLS},
                                pointstate::StructVector,
                                cache::MPCache{<: Any, <: Any, <: WLSValues},
                                coord_system::Symbol)
    P = polynomial(grid.shapefunction)
    point_to_grid!((grid.state.m, grid.state.w, grid.state.v, grid.state.f), cache) do it, p, i
        @_inline_meta
        @_propagate_inbounds_meta
        N = it.N
        ∇N = it.∇N
        w = it.w
        xₚ = pointstate.x[p]
        mₚ = pointstate.m[p]
        V0ₚ = pointstate.V0[p]
        Fₚ = pointstate.F[p]
        Cₚ = pointstate.C[p]
        σₚ = pointstate.σ[p]
        bₚ = pointstate.b[p]
        xᵢ = grid[i]
        m = mₚ * N
        v = w * Cₚ ⋅ P(xᵢ - xₚ)
        f = -(V0ₚ*det(Fₚ)) * stress_to_force(coord_system, N, ∇N, xₚ, σₚ) + m * bₚ
        m, w, v, f
    end
    @. grid.state.v /= grid.state.w
    grid
end

##################
# grid_to_point! #
##################

@generated function _grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector, N}}, shapevalues::ShapeValues, gridindices::Vector{<: Index}, p::Int) where {N}
    quote
        vals = tuple($([:(zero(eltype(pointstates[$i]))) for i in 1:N]...))
        @inbounds @simd for i in eachindex(shapevalues, gridindices)
            it = shapevalues[i]
            I = gridindices[i]
            res = g2p(it, I, p)
            vals = tuple($([:(vals[$i] + res[$i]) for i in 1:N]...))
        end
        $([:(setindex!(pointstates[$i], vals[$i], p)) for i in 1:N]...)
    end
end

function grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector}}, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    @assert all(==(npoints(cache)), length.(pointstates))
    pointmask !== nothing && @assert length(pointmask) == npoints(cache)
    @inbounds Threads.@threads for p in 1:npoints(cache)
        pointmask !== nothing && !pointmask[p] && continue
        _grid_to_point!(g2p, pointstates, cache.shapevalues[p], cache.gridindices[p], p)
    end
    pointstates
end

function grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector}}, grid::Grid{dim, T}, xₚ::AbstractVector; exclude = nothing) where {dim, T}
    @assert all(==(length(xₚ)), length.(pointstates))
    spat = sparsity_pattern(grid, xₚ; exclude)
    shapevalues_threads = [ShapeValues(T, grid.shapefunction) for _ in 1:Threads.nthreads()]
    gridindices_threads = [Index{dim}[] for _ in 1:Threads.nthreads()]
    Threads.@threads for p in 1:length(xₚ)
        x = xₚ[p]
        shapevalues = shapevalues_threads[Threads.threadid()]
        gridindices = gridindices_threads[Threads.threadid()]
        neighboring_nodes!(gridindices, grid, x, spat)
        update!(shapevalues, grid, x, gridindices)
        _grid_to_point!(g2p, pointstates, shapevalues, gridindices, p)
    end
    pointstates
end

function grid_to_point!(g2p, pointstate::AbstractVector, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    grid_to_point!((pointstate,), cache, pointmask) do it, I, p
        @_inline_meta
        @_propagate_inbounds_meta
        (g2p(it, I, p),)
    end
end

function grid_to_point!(g2p, pointstate::AbstractVector, grid::Grid, xₚ::AbstractVector; exclude = nothing)
    grid_to_point!((pointstate,), grid, xₚ; exclude) do it, I, p
        @_inline_meta
        @_propagate_inbounds_meta
        (g2p(it, I, p),)
    end
end

function velocity_gradient(coord_system::Symbol, x::Vec, v::Vec, ∇v::SecondOrderTensor{2})
    L = Poingr.Tensor3D(∇v)
    @inbounds coord_system == :axisymmetric ? L + @Mat([0 0 0; 0 0 0; 0 0 v[1]/x[1]]) : L
end

function default_grid_to_point!(pointstate::StructVector,
                                grid::Grid{dim, <: Any, <: WLS},
                                cache::MPCache{dim, <: Any, <: WLSValues},
                                dt::Real,
                                coord_system::Symbol) where {dim}
    P = polynomial(grid.shapefunction)
    p0 = P(zero(Vec{dim, Int}))
    ∇p0 = P'(zero(Vec{dim, Int}))
    grid_to_point!(pointstate.C, cache) do it, i, p
        @_inline_meta
        @_propagate_inbounds_meta
        w = it.w
        M⁻¹ = it.M⁻¹
        grid.state.v[i] ⊗ (w * M⁻¹ ⋅ P(grid[i] - pointstate.x[p]))
    end
    @inbounds Threads.@threads for p in eachindex(pointstate)
        Cₚ = pointstate.C[p]
        xₚ = pointstate.x[p]
        vₚ = Cₚ ⋅ p0
        ∇vₚ = velocity_gradient(coord_system, xₚ, vₚ, Cₚ ⋅ ∇p0)
        Fₚ = pointstate.F[p]
        pointstate.v[p] = vₚ
        pointstate.∇v[p] = ∇vₚ
        pointstate.F[p] = Fₚ + dt*(∇vₚ ⋅ Fₚ)
        pointstate.x[p] = xₚ + vₚ * dt
    end
    pointstate
end
