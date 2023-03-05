const BLOCKFACTOR = unsigned(3) # 2^3

struct BlockSpace{dim}
    particleindices::Vector{Int}
    # for blocks
    nparticles::Array{Int, dim}
    stops::Array{Int, dim}
    # for particles
    blockindices::Vector{Int}
    localindices::Vector{Int}
end
function BlockSpace(blocksize::Dims{dim}, npts::Int) where {dim}
    particleindices = Vector{Int}(undef, npts)
    nparticles = zeros(Int, blocksize)
    stops = zeros(Int, blocksize)
    blockindices = Vector{Int}(undef, npts)
    localindices = Vector{Int}(undef, npts)
    BlockSpace(particleindices, nparticles, stops, blockindices, localindices)
end

blocksize(bs::BlockSpace) = size(bs.nparticles)
num_particles(bs::BlockSpace, index...) = (@_propagate_inbounds_meta; bs.nparticles[index...])

function particleindices(bs::BlockSpace, index...)
    @boundscheck checkbounds(CartesianIndices(blocksize(bs)), index...)
    @inbounds begin
        stop = bs.stops[index...]
        start = stop - bs.nparticles[index...] + 1
        view(bs.particleindices, start:stop)
    end
end

function update!(bs::BlockSpace, lattice::Lattice, xₚ::AbstractVector)
    fillzero!(bs.nparticles)
    @inbounds for p in eachindex(xₚ)
        blk = sub2ind(blocksize(bs), whichblock(lattice, xₚ[p]))
        bs.blockindices[p] = blk
        bs.localindices[p] = iszero(blk) ? 0 : (bs.nparticles[blk] += 1)
    end
    cumsum!(vec(bs.stops), vec(bs.nparticles))
    @threaded_inbounds for p in eachindex(xₚ)
        blk = bs.blockindices[p]
        if !iszero(blk)
            i = bs.localindices[p]
            stop = bs.stops[blk]
            len = bs.nparticles[blk]
            bs.particleindices[stop-len+i] = p
        end
    end
end
sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
sub2ind(::Dims, ::Nothing)::Int = 0

# block-wise rough sparsity pattern
# CubicBSpline is still ok with rng=1
function update_sparsity_pattern!(sppat::AbstractArray{Bool}, bs::BlockSpace, rng::Int = 1)
    @assert blocksize(size(sppat)) ==  blocksize(bs)
    fillzero!(sppat)
    @inbounds for I in CartesianIndices(blocksize(bs))
        if !iszero(num_particles(bs, I))
            inds = neighbornodes_from_blockindex(size(sppat), I, rng)
            sppat[inds] .= true
        end
    end
end
@inline function neighbornodes_from_blockindex(gridsize::Dims, blk::CartesianIndex, i::Int)
    start = @. max((blk.I-1) << BLOCKFACTOR + 1 - i, 1)
    stop = @. min(blk.I << BLOCKFACTOR + 1 + i, gridsize)
    CartesianIndex(start):CartesianIndex(stop)
end

function reorder_particles!(particles::Particles, blkspace::BlockSpace)
    _reorder_particles!(particles, PointsInBlocksArray(blkspace))
end

struct PointsInBlocksArray{dim, T} <: AbstractArray{T, dim}
    parent::BlockSpace{dim}
end
function PointsInBlocksArray(parent::BlockSpace{dim}) where {dim}
    T = Base._return_type(particleindices, Tuple{typeof(parent), Int})
    PointsInBlocksArray{dim, T}(parent)
end
Base.parent(x::PointsInBlocksArray) = x.parent
Base.size(x::PointsInBlocksArray) = blocksize(parent(x))
Base.getindex(x::PointsInBlocksArray, index...) = particleindices(parent(x), index...)

####################
# block operations #
####################

blocksize(gridsize::Tuple{Vararg{Int}}) = (ncells = gridsize .- 1; @. (ncells - 1) >> BLOCKFACTOR + 1)
blocksize(lattice::Lattice) = blocksize(size(lattice))
blocksize(grid::Grid) = blocksize(size(grid))

"""
    Marble.whichblock(lattice, x::Vec)

Return block index where `x` locates.
The unit block size is `2^$BLOCKFACTOR` cells.

# Examples
```jldoctest
julia> lattice = Lattice(1, (0,10), (0,10))
11×11 Lattice{2, Float64}:
 [0.0, 0.0]   [0.0, 1.0]   [0.0, 2.0]   …  [0.0, 9.0]   [0.0, 10.0]
 [1.0, 0.0]   [1.0, 1.0]   [1.0, 2.0]      [1.0, 9.0]   [1.0, 10.0]
 [2.0, 0.0]   [2.0, 1.0]   [2.0, 2.0]      [2.0, 9.0]   [2.0, 10.0]
 [3.0, 0.0]   [3.0, 1.0]   [3.0, 2.0]      [3.0, 9.0]   [3.0, 10.0]
 [4.0, 0.0]   [4.0, 1.0]   [4.0, 2.0]      [4.0, 9.0]   [4.0, 10.0]
 [5.0, 0.0]   [5.0, 1.0]   [5.0, 2.0]   …  [5.0, 9.0]   [5.0, 10.0]
 [6.0, 0.0]   [6.0, 1.0]   [6.0, 2.0]      [6.0, 9.0]   [6.0, 10.0]
 [7.0, 0.0]   [7.0, 1.0]   [7.0, 2.0]      [7.0, 9.0]   [7.0, 10.0]
 [8.0, 0.0]   [8.0, 1.0]   [8.0, 2.0]      [8.0, 9.0]   [8.0, 10.0]
 [9.0, 0.0]   [9.0, 1.0]   [9.0, 2.0]      [9.0, 9.0]   [9.0, 10.0]
 [10.0, 0.0]  [10.0, 1.0]  [10.0, 2.0]  …  [10.0, 9.0]  [10.0, 10.0]

julia> Marble.whichblock(lattice, Vec(8.5, 1.5))
CartesianIndex(2, 1)
```
"""
@inline function whichblock(lattice::Lattice, x::Vec)
    I = whichcell(lattice, x)
    I === nothing && return nothing
    CartesianIndex(@. (I.I-1) >> BLOCKFACTOR + 1)
end

function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = AxisArray(nfill(1:2, Val(dim)))
    vec(map(st -> map(CartesianIndex{dim}, AxisArray(StepRange.(st, 2, blocksize)))::Array{CartesianIndex{dim}, dim}, starts))
end

# block-wise parallel computation
function parallel_each_particle(f, blkspace::BlockSpace)
    for blocks in threadsafe_blocks(blocksize(blkspace))
        blocks′ = filter(I -> !iszero(num_particles(blkspace, I)), blocks)
        @threaded_inbounds for blk in blocks′
            foreach(f, particleindices(blkspace, blk))
        end
    end
end
