struct MPCache{T, dim, F <: Interpolation, CS, MV <: MPValues{dim, T}}
    interp::F
    grid::Grid{T, dim, CS}
    spat::Array{Bool, dim}
    mpvalues::Vector{MV}
    ptsinblk::Array{Vector{Int}, dim}
    npts::Base.RefValue{Int}
    stamp::Base.RefValue{Float64}
end

# constructors
function MPCache(interp::Interpolation, grid::Grid{T, dim}, xₚ::AbstractVector{<: Vec{dim}}) where {dim, T}
    spat = fill(false, size(grid))
    npts = length(xₚ)
    mpvalues = [MPValues{dim, T}(interp) for _ in 1:npts]
    MPCache(interp, grid, spat, mpvalues, pointsinblock(grid, xₚ), Ref(npts), Ref(NaN))
end
MPCache(interp::Interpolation, grid::Grid, pointstate::AbstractVector) = MPCache(interp, grid, pointstate.x)

# helper functions
gridsize(cache::MPCache) = size(cache.grid)
num_points(cache::MPCache) = cache.npts[]
get_pointsinblock(cache::MPCache) = cache.ptsinblk
get_sparsitypattern(cache::MPCache) = cache.spat
get_interpolation(cache::MPCache) = cache.interp
get_stamp(cache::MPCache) = cache.stamp[]

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
reorder_pointstate!(pointstate::AbstractVector, cache::MPCache) = reorder_pointstate!(pointstate, get_pointsinblock(cache))

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

function update_sparsitypattern!(spat::Array{Bool}, interp::Interpolation, grid::Grid, pointstate::AbstractVector, ptsinblk::AbstractArray{Vector{Int}}; exclude)
    @assert size(spat) == size(grid)
    fill!(spat, false)
    for blocks in threadsafe_blocks(size(grid))
        Threads.@threads for blockindex in blocks
            for p in ptsinblk[blockindex]
                inds = neighbornodes(interp, grid, LazyRow(pointstate, p))
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

function update!(cache::MPCache, pointstate; exclude::Union{Nothing, AbstractArray{Bool}} = nothing)
    grid = cache.grid
    mpvalues = cache.mpvalues
    ptsinblk = cache.ptsinblk
    spat = cache.spat

    cache.npts[]  = length(pointstate)
    cache.stamp[] = time()
    allocate!(i -> eltype(mpvalues)(), mpvalues, length(pointstate))

    pointsinblock!(ptsinblk, grid, pointstate.x)
    update_sparsitypattern!(spat, get_interpolation(cache), grid, pointstate, ptsinblk; exclude)

    Threads.@threads for p in 1:length(pointstate)
        @inbounds update!(mpvalues[p], grid, LazyRow(pointstate, p), spat)
    end

    cache
end

function update_sparsitypattern!(spat::SpArray, mpcache::MPCache)
    update_sparsitypattern!(spat, get_sparsitypattern(mpcache))
    spat.stamp[] = get_stamp(mpcache)
    spat
end

function eachpoint_blockwise_parallel(f, cache::MPCache)
    for blocks in threadsafe_blocks(gridsize(cache))
        Threads.@threads for blockindex in blocks
            @inbounds for p in get_pointsinblock(cache)[blockindex]
                f(p)
            end
        end
    end
end

##################
# point_to_grid! #
##################

function check_gridstate(gridstate::AbstractArray, cache::MPCache)
    @assert size(gridstate) == gridsize(cache)
end
function check_gridstate(gridstate::SpArray, cache::MPCache)
    @assert size(gridstate) == gridsize(cache)
    @assert get_stamp(gridstate) == get_stamp(cache)
end

function point_to_grid!(p2g, gridstates, mps::MPValues)
    @_inline_propagate_inbounds_meta
    @simd for i in 1:length(mps)
        I = gridindices(mps, i)
        map_tuple(add!, gridstates, p2g(mps[i], I), I)
    end
end

function point_to_grid!(p2g, gridstates, cache::MPCache; zeroinit::Bool = true)
    map_tuple(check_gridstate, gridstates, cache)
    zeroinit && map_tuple(fillzero!, gridstates)
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

function point_to_grid!(p2g, gridstates, cache::MPCache, pointmask::AbstractVector{Bool}; zeroinit::Bool = true)
    map_tuple(check_gridstate, gridstates, cache)
    @assert length(pointmask) == num_points(cache)
    zeroinit && map_tuple(fillzero!, gridstates)
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

function check_pointstate(pointstate::AbstractVector, cache::MPCache)
    @assert length(pointstate) == num_points(cache)
end

function grid_to_point(g2p, mps::MPValues)
    @_inline_propagate_inbounds_meta
    vals = g2p(first(mps), gridindices(mps, 1))
    @simd for i in 2:length(mps)
        I = gridindices(mps, i)
        vals = map_tuple(+, vals, g2p(mps[i], I))
    end
    vals
end

function grid_to_point(g2p, cache::MPCache)
    LazyDotArray(1:num_points(cache)) do p
        @_inline_propagate_inbounds_meta
        grid_to_point(
            (mp, I) -> (@_inline_propagate_inbounds_meta; g2p(mp, I, p)),
            cache.mpvalues[p]
        )
    end
end

function grid_to_point!(g2p, pointstates, cache::MPCache)
    map_tuple(check_pointstate, pointstates, cache)
    results = grid_to_point(g2p, cache)
    Threads.@threads for p in 1:num_points(cache)
        @inbounds map_tuple(setindex!, pointstates, results[p], p)
    end
end

function grid_to_point!(g2p, pointstates, cache::MPCache, pointmask::AbstractVector{Bool})
    map_tuple(check_pointstate, pointstates, cache)
    @assert length(pointmask) == num_points(cache)
    results = grid_to_point(g2p, cache)
    Threads.@threads for p in 1:num_points(cache)
        @inbounds pointmask[p] && map_tuple(setindex!, pointstates, results[p], p)
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

function smooth_pointstate!(vals::AbstractVector, Vₚ::AbstractVector, gridstate::AbstractArray, cache::MPCache)
    @assert length(vals) == length(Vₚ) == num_points(cache)
    grid = cache.grid
    basis = PolynomialBasis{1}()
    point_to_grid!((gridstate.poly_coef, gridstate.poly_mat), cache) do mp, p, i
        @_inline_propagate_inbounds_meta
        P = value(basis, mp.xp - grid[i])
        VP = (mp.N * Vₚ[p]) * P
        VP * vals[p], VP ⊗ P
    end
    @dot_threads gridstate.poly_coef = safe_inv(gridstate.poly_mat) ⋅ gridstate.poly_coef
    grid_to_point!(vals, cache) do mp, i, p
        @_inline_propagate_inbounds_meta
        P = value(basis, mp.xp - grid[i])
        mp.N * (P ⋅ gridstate.poly_coef[i])
    end
end
