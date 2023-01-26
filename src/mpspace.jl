struct MPSpace{dim, T, F <: Interpolation, C <: CoordinateSystem, V <: MPValue{dim, T}}
    interp::F
    grid::Grid{dim, T, C}
    sppat::Array{Bool, dim}
    mpvals::Vector{V}
    nodeinds::Vector{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}}
    ptspblk::Array{PushVector{Int}, dim}
    npts::RefValue{Int}
    stamp::RefValue{Float64}
end

# constructors
function MPSpace(interp::Interpolation, grid::Grid{dim, T}, xₚ::AbstractVector{<: Vec}) where {dim, T}
    sppat = fill(false, size(grid))
    npts = length(xₚ)
    mpvals = [MPValue{dim, T}(interp) for _ in 1:npts]
    nodeinds = [CartesianIndices(nfill(1:0, Val(dim))) for _ in 1:npts]
    MPSpace(interp, grid, sppat, mpvals, nodeinds, pointsperblock(grid, xₚ), Ref(npts), Ref(NaN))
end
MPSpace(interp::Interpolation, grid::Grid, pointstate::AbstractVector) = MPSpace(interp, grid, pointstate.x)

# helper functions
gridsize(space::MPSpace) = size(space.grid)
num_points(space::MPSpace) = space.npts[]
get_mpvalue(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.mpvals[i])
get_nodeindices(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.nodeinds[i])
get_interpolation(space::MPSpace) = space.interp
get_grid(space::MPSpace) = space.grid
get_sppat(space::MPSpace) = space.sppat
get_pointsperblock(space::MPSpace) = space.ptspblk
get_stamp(space::MPSpace) = space.stamp[]

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
function pointsperblock!(ptspblk::AbstractArray{PushVector{Int}}, grid::Grid, xₚ::AbstractVector)
    empty!.(ptspblk)
    @inbounds for p in 1:length(xₚ)
        I = whichblock(grid, xₚ[p])
        I === nothing || push!(ptspblk[I], p)
    end
    finish!.(ptspblk)
    ptspblk
end
function pointsperblock(grid::Grid, xₚ::AbstractVector)
    ptspblk = Array{PushVector{Int}}(undef, blocksize(size(grid)))
    @inbounds for i in eachindex(ptspblk)
        ptspblk[i] = PushVector{Int}()
    end
    pointsperblock!(ptspblk, grid, xₚ)
end

function allocate!(f, x::Vector, n::Integer)
    len = length(x)
    if n > len # growend
        resize!(x, n)
        for i in len+1:n
            @inbounds x[i] = f(i)
        end
    end
    x
end

function update!(space::MPSpace{dim, T}, pointstate::AbstractVector; exclude::Union{Nothing, AbstractArray{Bool}} = nothing) where {dim, T}
    space.npts[]  = length(pointstate)
    space.stamp[] = time()

    allocate!(space.mpvals, length(pointstate)) do i
        interp = get_interpolation(space)
        MPValue{dim, T}(interp)
    end

    update_sparsity_pattern!(space, pointstate; exclude)
    @inbounds Threads.@threads for p in 1:length(pointstate)
        # `space.nodeinds` had been updated in `update_sparsity_pattern`
        update!(get_mpvalue(space, p),
                get_grid(space),
                get_sppat(space),
                get_nodeindices(space, p),
                LazyRow(pointstate, p))
    end

    space
end

function update_sparsity_pattern!(space::MPSpace, pointstate::AbstractVector; exclude::Union{Nothing, AbstractArray{Bool}} = nothing)
    @assert length(pointstate) == num_points(space)

    # update `pointsperblock`
    pointsperblock!(get_pointsperblock(space), get_grid(space), pointstate.x)

    # update sparsity pattern
    # `space.nodeinds` is also updated
    sppat = fillzero!(get_sppat(space))
    eachpoint_blockwise_parallel(space) do p
        @inbounds begin
            inds = neighbornodes(get_interpolation(space), get_grid(space), LazyRow(pointstate, p))
            space.nodeinds[p] = inds
            sppat[inds] .= true
        end
    end

    # handle excluded domain
    if exclude !== nothing
        @. sppat &= !exclude
        eachpoint_blockwise_parallel(space) do p
            @inbounds begin
                inds = neighbornodes(get_grid(space), pointstate.x[p], 1)
                sppat[inds] .= true
            end
        end
    end

    sppat
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
