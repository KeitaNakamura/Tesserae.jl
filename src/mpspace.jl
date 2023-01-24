struct MPSpace{dim, T, F <: Interpolation, C <: CoordinateSystem, V <: MPValue{dim, T}}
    interp::F
    grid::Grid{dim, T, C}
    sppat::Array{Bool, dim}
    mpvals::Vector{V}
    ptspblk::Array{Vector{Int}, dim}
    npts::RefValue{Int}
    stamp::RefValue{Float64}
end

# constructors
function MPSpace(interp::Interpolation, grid::Grid{dim, T}, xₚ::AbstractVector{<: Vec{dim}}) where {dim, T}
    sppat = fill(false, size(grid))
    npts = length(xₚ)
    mpvals = [MPValue{dim, T}(interp) for _ in 1:npts]
    MPSpace(interp, grid, sppat, mpvals, pointsperblock(grid, xₚ), Ref(npts), Ref(NaN))
end
MPSpace(interp::Interpolation, grid::Grid, pointstate::AbstractVector) = MPSpace(interp, grid, pointstate.x)

# helper functions
gridsize(space::MPSpace) = size(space.grid)
num_points(space::MPSpace) = space.npts[]
mpvalue(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.mpvals[i])
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
    @inbounds @simd for i in eachindex(ptspblk)
        ptspblk[i] = Int[]
    end
    pointsperblock!(ptspblk, grid, xₚ)
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

function update!(space::MPSpace{dim, T}, pointstate::AbstractVector; exclude::Union{Nothing, AbstractArray{Bool}} = nothing) where {T, dim}
    space.npts[]  = length(pointstate)
    space.stamp[] = time()

    allocate!(space.mpvals, length(pointstate)) do i
        interp = get_interpolation(space)
        MPValue{dim, T}(interp)
    end

    update_sparsity_pattern!(space, pointstate; exclude)
    @inbounds Threads.@threads for p in 1:length(pointstate)
        mp = mpvalue(space, p)
        update!(mp, get_grid(space), LazyRow(pointstate, p), get_sppat(space))
    end

    space
end

function update_sparsity_pattern!(space::MPSpace, pointstate::AbstractVector; exclude::Union{Nothing, AbstractArray{Bool}} = nothing)
    # update `pointsperblock`
    pointsperblock!(get_pointsperblock(space), get_grid(space), pointstate.x)

    # update sparsity pattern
    sppat = get_sppat(space)
    fill!(sppat, false)
    eachpoint_blockwise_parallel(space) do p
        @_inline_propagate_inbounds_meta
        inds = nodeindices(get_interpolation(space), get_grid(space), LazyRow(pointstate, p))
        sppat[inds] .= true
    end

    # handle excluded domain
    if exclude !== nothing
        @. sppat &= !exclude
        eachpoint_blockwise_parallel(space) do p
            @_inline_propagate_inbounds_meta
            inds = nodeindices(get_grid(space), pointstate.x[p], 1)
            sppat[inds] .= true
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
