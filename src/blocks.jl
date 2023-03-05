const BLOCKFACTOR = unsigned(3) # 2^3

struct ParticlesInBlocks{T, dim} <: AbstractArray{T, dim}
    particleindices::Vector{Int}
    # for blocks
    counts::Array{Int, dim}
    stops::Array{Int, dim}
    # for particles
    blockindices::Vector{Int}
    localindices::Vector{Int}
end
function ParticlesInBlocks(blocksize::Dims{dim}, npts::Int) where {dim}
    particleindices = Vector{Int}(undef, npts)
    counts = zeros(Int, blocksize)
    stops = zeros(Int, blocksize)
    blockindices = Vector{Int}(undef, npts)
    localindices = Vector{Int}(undef, npts)
    T = typeof(view(particleindices, 1:0))
    ParticlesInBlocks{T, dim}(particleindices, counts, stops, blockindices, localindices)
end

Base.IndexStyle(::Type{<: ParticlesInBlocks}) = IndexLinear()
Base.size(pb::ParticlesInBlocks) = size(pb.counts)
function Base.getindex(pb::ParticlesInBlocks, i::Integer)
    @boundscheck checkbounds(pb, i)
    @inbounds begin
        stop = pb.stops[i]
        start = stop - pb.counts[i] + 1
        view(pb.particleindices, start:stop)
    end
end

function update_sparsity_pattern!(pb::ParticlesInBlocks, lattice::Lattice, xₚ::AbstractVector)
    fillzero!(pb.counts)
    @inbounds for p in eachindex(xₚ)
        blk = sub2ind(size(pb), whichblock(lattice, xₚ[p]))
        pb.blockindices[p] = blk
        pb.localindices[p] = iszero(blk) ? 0 : (pb.counts[blk] += 1)
    end
    cumsum!(vec(pb.stops), vec(pb.counts))
    @threaded_inbounds for p in eachindex(xₚ)
        blk = pb.blockindices[p]
        if !iszero(blk)
            i = pb.localindices[p]
            stop = pb.stops[blk]
            len = pb.counts[blk]
            pb.particleindices[stop-len+i] = p
        end
    end
end
sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
sub2ind(::Dims, ::Nothing)::Int = 0

# block-wise rough sparsity pattern
# CubicBSpline is still ok with rng=1
function update_sparsity_pattern!(sppat::AbstractArray{Bool}, pb::ParticlesInBlocks, rng::Int = 1)
    @assert blocksize(size(sppat)) ==  size(pb)
    fillzero!(sppat)
    @inbounds for I in CartesianIndices(pb)
        if !isempty(pb[I])
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
