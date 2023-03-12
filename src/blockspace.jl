const BLOCKFACTOR = unsigned(3) # 2^3

struct ParticlesInBlocksArray{dim, T, Tinds <: AbstractVector{Int}, Tstops <: AbstractArray{Int, dim}} <: AbstractArray{T, dim}
    particleindices::Tinds
    stops::Tstops
end

function ParticlesInBlocksArray(particleindices::AbstractVector, stops::AbstractArray)
    dim = ndims(stops)
    T = typeof(view(particleindices, 1:1))
    ParticlesInBlocksArray{dim, T, typeof(particleindices), typeof(stops)}(particleindices, stops)
end

function ParticlesInBlocksArray(blocksize::Dims{dim}, npts::Int) where {dim}
    particleindices = Vector{Int}(undef, npts)
    stops = zeros(Int, blocksize)
    ParticlesInBlocksArray(particleindices, stops)
end

Base.IndexStyle(::Type{<: ParticlesInBlocksArray}) = IndexLinear()
Base.size(x::ParticlesInBlocksArray) = size(x.stops)
@inline function Base.getindex(x::ParticlesInBlocksArray, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds begin
        stop = x.stops[i]
        start = i==1 ? 1 : x.stops[i-1]+1
        view(x.particleindices, start:stop)
    end
end

struct BlockSpace{dim, BA <: ParticlesInBlocksArray{dim}}
    blkarray::BA
    nparticles::Vector{Array{Int, dim}}
    blockindices::Vector{Int}
    localindices::Vector{Int}
end
function BlockSpace(blocksize::Dims, npts::Int)
    blkarray = ParticlesInBlocksArray(blocksize, npts)
    nparticles = [zeros(Int, blocksize) for _ in 1:Threads.nthreads()]
    blockindices = Vector{Int}(undef, npts)
    localindices = Vector{Int}(undef, npts)
    BlockSpace(blkarray, nparticles, blockindices, localindices)
end

blocksize(bs::BlockSpace) = size(bs.blkarray)
num_particles(bs::BlockSpace, index...) = (@_propagate_inbounds_meta; last(bs.nparticles)[index...])

@inline function particleindices(bs::BlockSpace, index...)
    @_propagate_inbounds_meta
    bs.blkarray[index...]
end

function update!(bs::BlockSpace, lattice::Lattice, xₚ::AbstractVector)
    blkarray = bs.blkarray
    fillzero!.(bs.nparticles)
    @threaded_inbounds :static for p in eachindex(xₚ)
        id = Threads.threadid()
        blk = sub2ind(blocksize(bs), whichblock(lattice, xₚ[p]))
        bs.blockindices[p] = blk
        bs.localindices[p] = iszero(blk) ? 0 : (bs.nparticles[id][blk] += 1)
    end
    for i in 1:Threads.nthreads()-1
        @inbounds broadcast!(+, bs.nparticles[i+1], bs.nparticles[i+1], bs.nparticles[i])
    end
    nptsinblks = last(bs.nparticles)
    cumsum!(vec(blkarray.stops), vec(nptsinblks))
    @threaded_inbounds :static for p in eachindex(xₚ)
        blk = bs.blockindices[p]
        if !iszero(blk)
            id = Threads.threadid()
            offset = id==1 ? 0 : bs.nparticles[id-1][blk]
            i = offset + bs.localindices[p]
            stop = blkarray.stops[blk]
            len = nptsinblks[blk]
            blkarray.particleindices[stop-len+i] = p
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
    _reorder_particles!(particles, blkspace.blkarray)
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

# block-wise parallel computation
function parallel_each_particle(f, blkspace::BlockSpace)
    for blocks in threadsafe_blocks(blocksize(blkspace))
        blocks′ = filter(I -> !iszero(num_particles(blkspace, I)), blocks)
        @threaded_inbounds for blk in blocks′
            foreach(f, particleindices(blkspace, blk))
        end
    end
end
