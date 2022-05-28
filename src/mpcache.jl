struct MPCache{dim, T, Tmp <: MPValues{dim, T}}
    mpvalues::Vector{Tmp}
    gridsize::NTuple{dim, Int}
    npoints::Base.RefValue{Int}
    pointsinblock::Array{Vector{Int}, dim}
    spat::Array{Bool, dim}
end

# constructors
function MPCache(grid::Grid{T, dim}, xₚ::AbstractVector{<: Vec{dim}}) where {dim, T}
    check_interpolation(grid)
    npoints = length(xₚ)
    mpvalues = [MPValues{dim, T}(grid.interpolation) for _ in 1:npoints]
    MPCache(mpvalues, size(grid), Ref(npoints), pointsinblock(grid, xₚ), fill(false, size(grid)))
end
MPCache(grid::Grid, pointstate::AbstractVector) = MPCache(grid, pointstate.x)

# helper functions
gridsize(cache::MPCache) = cache.gridsize
npoints(cache::MPCache) = cache.npoints[]
pointsinblock(cache::MPCache) = cache.pointsinblock

# reorder_pointstate!
function reorder_pointstate!(pointstate::AbstractVector, ptsinblk::Array)
    @assert length(pointstate) == sum(length, ptsinblk)
    inds = Vector{Int}(undef, length(pointstate))
    cnt = 1
    for blocks in threadsafe_blocks(@. $size(ptsinblk) << BLOCK_UNIT + 1)
        @inbounds for blockindex in blocks
            block = ptsinblk[blockindex]
            for i in eachindex(block)
                inds[cnt] = block[i]
                block[i] = cnt
                cnt += 1
            end
        end
    end
    @inbounds @. pointstate = pointstate[inds]
    pointstate
end
reorder_pointstate!(pointstate::AbstractVector, cache::MPCache) = reorder_pointstate!(pointstate, pointsinblock(cache))

# pointsinblock!
function pointsinblock!(ptsinblk::AbstractArray{Vector{Int}}, grid::Grid, xₚ::AbstractVector)
    empty!.(ptsinblk)
    @inbounds for p in 1:length(xₚ)
        I = whichblock(grid, xₚ[p])
        I === nothing || push!(ptsinblk[I], p)
    end
    ptsinblk
end
function pointsinblock(grid::Grid, xₚ::AbstractVector)
    ptsinblk = Array{Vector{Int}}(undef, blocksize(grid))
    @inbounds @simd for i in eachindex(ptsinblk)
        ptsinblk[i] = Int[]
    end
    pointsinblock!(ptsinblk, grid, xₚ)
end

function sparsity_pattern!(spat::Array{Bool}, grid::Grid, pointstate::AbstractVector, ptsinblk::AbstractArray{Vector{Int}}; exclude)
    @assert size(spat) == size(grid)
    fill!(spat, false)
    for blocks in threadsafe_blocks(size(grid))
        Threads.@threads for blockindex in blocks
            for p in ptsinblk[blockindex]
                inds = neighbornodes(grid.interpolation, grid, LazyRow(pointstate, p))
                @inbounds spat[inds] .= true
            end
        end
    end
    if exclude !== nothing
        @. spat &= !exclude
        for blocks in threadsafe_blocks(size(grid))
            Threads.@threads for blockindex in blocks
                for p in ptsinblk[blockindex]
                    inds = neighbornodes(grid, pointstate.x[p], 1)
                    @inbounds spat[inds] .= true
                end
            end
        end
    end
    spat
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

function update!(cache::MPCache, grid::Grid, pointstate; exclude::Union{Nothing, AbstractArray{Bool}} = nothing)
    @assert size(grid) == gridsize(cache)

    mpvalues = cache.mpvalues
    pointsinblock = cache.pointsinblock
    spat = cache.spat

    cache.npoints[] = length(pointstate)
    allocate!(i -> eltype(mpvalues)(), mpvalues, length(pointstate))

    pointsinblock!(pointsinblock, grid, pointstate.x)
    sparsity_pattern!(spat, grid, pointstate, pointsinblock; exclude)

    Threads.@threads for p in 1:length(pointstate)
        @inbounds update!(mpvalues[p], grid, LazyRow(pointstate, p), spat)
    end

    gridstate = grid.state
    copyto!(gridstate.spat, spat)
    reinit!(gridstate)

    cache
end

function eachpoint_blockwise_parallel(f, cache::MPCache)
    for blocks in threadsafe_blocks(gridsize(cache))
        Threads.@threads for blockindex in blocks
            @inbounds for p in pointsinblock(cache)[blockindex]
                f(p)
            end
        end
    end
end

##################
# point_to_grid! #
##################

checksize(xs, dims) = @assert all(maptuple(x -> size(x) == dims, xs))

function point_to_grid!(p2g, gridstates, mps::MPValues)
    @_inline_propagate_inbounds_meta
    @simd for i in 1:length(mps)
        I = gridindices(mps, i)
        maptuple(add!, gridstates, p2g(mps[i], I), I)
    end
end

function point_to_grid!(p2g, gridstates, cache::MPCache; zeroinit::Bool = true)
    checksize(gridstates, gridsize(cache))
    zeroinit && maptuple(fillzero!, gridstates)
    eachpoint_blockwise_parallel(cache) do p
        @_inline_propagate_inbounds_meta
        point_to_grid!(
            (mp, I) -> (@_inline_propagate_inbounds_meta; p2g(mp, p, I)),
            gridstates,
            cache.mpvalues[p],
        )
    end
    gridstates
end

function point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray}}, cache::MPCache, pointmask::AbstractVector{Bool}; zeroinit::Bool = true)
    checksize(gridstates, gridsize(cache))
    @assert length(pointmask) == npoints(cache)
    zeroinit && maptuple(fillzero!, gridstates)
    eachpoint_blockwise_parallel(cache) do p
        @_inline_propagate_inbounds_meta
        pointmask[p] && point_to_grid!(
            (mp, I) -> (@_inline_propagate_inbounds_meta; p2g(mp, p, I)),
            gridstates,
            cache.mpvalues[p],
        )
    end
    gridstates
end

##################
# grid_to_point! #
##################

function grid_to_point(g2p, mps::MPValues)
    @_inline_propagate_inbounds_meta
    vals = g2p(first(mps), gridindices(mps, 1))
    @simd for i in 2:length(mps)
        I = gridindices(mps, i)
        vals = maptuple(+, vals, g2p(mps[i], I))
    end
    vals
end

function grid_to_point(g2p, cache::MPCache)
    LazyDotArray(1:npoints(cache)) do p
        @_inline_propagate_inbounds_meta
        grid_to_point(
            (mp, I) -> (@_inline_propagate_inbounds_meta; g2p(mp, I, p)),
            cache.mpvalues[p]
        )
    end
end

function grid_to_point!(g2p, pointstates, cache::MPCache)
    checksize(pointstates, (npoints(cache),))
    results = grid_to_point(g2p, cache)
    Threads.@threads for p in 1:npoints(cache)
        @inbounds maptuple(setindex!, pointstates, results[p], p)
    end
end

function grid_to_point!(g2p, pointstates, cache::MPCache, pointmask::AbstractVector{Bool})
    checksize(pointstates, (npoints(cache),))
    @assert length(pointmask) == npoints(cache)
    results = grid_to_point(g2p, cache)
    Threads.@threads for p in 1:npoints(cache)
        @inbounds pointmask[p] && maptuple(setindex!, pointstates, results[p], p)
    end
end

######################
# smooth_pointstate! #
######################

@generated function safe_inv(x::Mat{dim, dim, T, L}) where {dim, T, L}
    exps = fill(:z, L-1)
    quote
        @_inline_meta
        z = zero(T)
        isapproxzero(det(x)) ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x)
        # Tensorial.rank(x) != dim ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x) # this is very slow but stable
    end
end

function smooth_pointstate!(vals::AbstractVector, Vₚ::AbstractVector, grid::Grid, cache::MPCache)
    @assert length(vals) == length(Vₚ) == npoints(cache)
    basis = PolynomialBasis{1}()
    point_to_grid!((grid.state.poly_coef, grid.state.poly_mat), cache) do mp, p, i
        @_inline_propagate_inbounds_meta
        P = value(basis, mp.xp - grid[i])
        VP = (mp.N * Vₚ[p]) * P
        VP * vals[p], VP ⊗ P
    end
    @dot_threads grid.state.poly_coef = safe_inv(grid.state.poly_mat) ⋅ grid.state.poly_coef
    grid_to_point!(vals, cache) do mp, i, p
        @_inline_propagate_inbounds_meta
        P = value(basis, mp.xp - grid[i])
        mp.N * (P ⋅ grid.state.poly_coef[i])
    end
end
