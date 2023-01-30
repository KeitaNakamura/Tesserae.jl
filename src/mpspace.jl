mutable struct MPSpace{dim, T, C <: CoordinateSystem, I <: Interpolation, MP <: MPValue{dim, T, I}}
    grid::Grid{dim, T, C}
    sppat::Array{Bool, dim}
    mpvals::Vector{MP}
    nodeinds::Vector{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}}
    ptspblk::Array{Vector{Int}, dim}
    stamp::Float64
end

# constructors
function MPSpace(interp::Interpolation, grid::Grid{dim, T}, xₚ::AbstractVector{<: Vec}) where {dim, T}
    sppat = fill(false, size(grid))
    npts = length(xₚ)
    mpvals = [MPValue{dim, T}(interp) for _ in 1:npts]
    nodeinds = [CartesianIndices(nfill(1:0, Val(dim))) for _ in 1:npts]
    MPSpace(grid, sppat, mpvals, nodeinds, pointsperblock(grid, xₚ), NaN)
end
MPSpace(interp::Interpolation, grid::Grid, pointstate::AbstractVector) = MPSpace(interp, grid, pointstate.x)

# helper functions
gridsize(space::MPSpace) = size(space.grid)
num_points(space::MPSpace) = length(space.mpvals)
get_mpvalue(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.mpvals[i])
get_nodeindices(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.nodeinds[i])
get_grid(space::MPSpace) = space.grid
get_sppat(space::MPSpace) = space.sppat
get_pointsperblock(space::MPSpace) = space.ptspblk
get_stamp(space::MPSpace) = space.stamp

# reorder_pointstate!
function reorder_pointstate!(pointstate::AbstractVector, ptspblk::Array)
    inds = Vector{Int}(undef, sum(length, ptspblk))
    cnt = 1
    for blocks in threadsafe_blocks(size(ptspblk))
        @inbounds for blockindex in blocks
            block = ptspblk[blockindex]
            for i in eachindex(block)
                inds[cnt] = block[i]
                block[i] = cnt
                cnt += 1
            end
        end
    end
    rest = pointstate[setdiff(1:length(pointstate), inds)]
    @inbounds begin
        pointstate[1:length(inds)] .= view(pointstate, inds)
        pointstate[length(inds)+1:end] .= rest
    end
    pointstate
end
reorder_pointstate!(pointstate::AbstractVector, space::MPSpace) = reorder_pointstate!(pointstate, get_pointsperblock(space))

# pointsperblock!
function pointsperblock!(ptspblk::AbstractArray{Vector{Int}}, grid::Grid, xₚ::AbstractVector)
    empty!.(ptspblk)
    @inbounds for p in 1:length(xₚ)
        I = whichblock(grid, xₚ[p])
        I === nothing || push!(ptspblk[I], p)
    end
    ptspblk
end
function pointsperblock(grid::Grid, xₚ::AbstractVector)
    ptspblk = Array{Vector{Int}}(undef, blocksize(size(grid)))
    @inbounds for i in eachindex(ptspblk)
        ptspblk[i] = Int[]
    end
    pointsperblock!(ptspblk, grid, xₚ)
end
function update_pointsperblock!(space::MPSpace, xₚ::AbstractVector)
    pointsperblock!(get_pointsperblock(space), get_grid(space), xₚ)
end

function update!(space::MPSpace{dim, T}, pointstate::AbstractVector; filter::Union{Nothing, AbstractArray{Bool}} = nothing) where {dim, T}
    @assert num_points(space) == length(pointstate)

    space.stamp = time()
    update_pointsperblock!(space, pointstate.x)
    update_sparsity_pattern!(space)
    update_mpvalues!(space, pointstate, filter)

    space
end

@inline function neighbornodes_from_blockindex(gridsize::Dims, blk::CartesianIndex, i::Int)
    lo = blk.I .- i
    hi = blk.I .+ i
    start = @. max((lo-1) << BLOCK_UNIT + 1, 1)
    stop = @. min(hi << BLOCK_UNIT + 1, gridsize)
    CartesianIndex(start):CartesianIndex(stop)
end
function update_sparsity_pattern!(space::MPSpace)
    sppat = fillzero!(get_sppat(space))
    ptspblk = get_pointsperblock(space)
    @inbounds for I in CartesianIndices(ptspblk)
        blk = ptspblk[I]
        if !isempty(blk)
            inds = neighbornodes_from_blockindex(gridsize(space), I, 1)
            sppat[inds] .= true
        end
    end
end

function update_mpvalues!(space::MPSpace, pointstate::AbstractVector, filter::Union{Nothing, AbstractArray{Bool}})
    # following fields are updated in this function
    # * `space.mpvals`
    # * `space.nodeinds`

    @assert length(pointstate) == num_points(space)

    grid = get_grid(space)
    @inbounds Threads.@threads for p in 1:num_points(space)
        mp = get_mpvalue(space, p)
        pt = LazyRow(pointstate, p)
        space.nodeinds[p] = neighbornodes(mp, grid, pt)
        # normally update mpvalues here
        filter===nothing && update!(get_mpvalue(space, p), grid, AllTrue(), get_nodeindices(space, p), pt)
    end

    # handle excluded domain
    if filter !== nothing
        sppat = get_sppat(space)
        @. sppat &= filter
        eachpoint_blockwise_parallel(space) do p
            @inbounds begin
                inds = neighbornodes(grid, pointstate.x[p], 1)
                sppat[inds] .= true
            end
        end
        # update mpvalues after completely updating `sppat`
        @inbounds Threads.@threads for p in 1:num_points(space)
            update!(get_mpvalue(space, p), grid, sppat, get_nodeindices(space, p), LazyRow(pointstate, p))
        end
    end

    space
end

function update_sparsity_pattern!(gridstate::SpArray, space::MPSpace)
    @assert is_parent(gridstate)
    @assert size(gridstate) == gridsize(space)
    update_sparsity_pattern!(gridstate, get_sppat(space))
    set_stamp!(gridstate, get_stamp(space))
    gridstate
end

function eachpoint_blockwise_parallel(f, space::MPSpace)
    for blocks in threadsafe_blocks(blocksize(gridsize(space)))
        pointsperblock = collect(Iterators.filter(!isempty, Iterators.map(blkidx->get_pointsperblock(space)[blkidx], blocks)))
        Threads.@threads for pointindices in pointsperblock
            foreach(f, pointindices)
        end
    end
end
