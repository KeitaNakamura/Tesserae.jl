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

sub2ind(dims::Dims, I)::Int = @inbounds LinearIndices(dims)[I]
sub2ind(::Dims, ::Nothing)::Int = 0
function update!(pb::ParticlesInBlocks, lattice::Lattice, xₚ::AbstractVector)
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

struct MPSpace{dim, T, It <: Interpolation, V, VI, PB <: ParticlesInBlocks, GS <: Union{Trues, SpPattern}}
    interp::It
    mpvals::MPValues{dim, T, V, VI}
    ptsinblks::PB
    sppat::Array{Bool, dim}
    gridsppat::GS # sppat used in SpGrid
end

# constructors
function MPSpace(itp::Interpolation, lattice::Lattice{dim, T}, xₚ::AbstractVector{<: Vec{dim}}, gridsppat) where {dim, T}
    npts = length(xₚ)
    gridsize = size(lattice)
    mpvals = MPValues{dim, T}(itp, npts)
    ptsinblks = ParticlesInBlocks(blocksize(gridsize), npts)
    sppat = fill(false, gridsize)
    MPSpace(itp, mpvals, ptsinblks, sppat, gridsppat)
end
MPSpace(itp::Interpolation, grid::Grid, particles::Particles) = MPSpace(itp, get_lattice(grid), particles.x, get_sppat(grid))

# helper functions
gridsize(space::MPSpace) = size(space.sppat)
num_particles(space::MPSpace) = length(space.mpvals)
get_interpolation(space::MPSpace) = space.interp
get_particlesinblocks(space::MPSpace) = space.ptsinblks
get_sppat(space::MPSpace) = space.sppat
get_gridsppat(space::MPSpace) = space.gridsppat

reorder_particles!(particles::Particles, space::MPSpace) = _reorder_particles!(particles, get_particlesinblocks(space))

# values
Base.values(space::MPSpace) = space.mpvals
Base.values(space::MPSpace, i::Integer) = (@_propagate_inbounds_meta; values(space.mpvals, i))
# set/get gridindices
@inline function neighbornodes(space::MPSpace, i::Integer)
    @_propagate_inbounds_meta
    @inbounds begin
        inds = neighbornodes(space.mpvals, i)
        nonzeroindices(space, inds)
    end
end
# nonzeroindices
struct NonzeroIndices{I, dim, A <: AbstractArray{I, dim}} <: AbstractArray{NonzeroIndex{I}, dim}
    parent::A
    nzinds::Array{Int, dim}
end
Base.size(x::NonzeroIndices) = size(x.parent)
@inline function Base.getindex(x::NonzeroIndices, I...)
    @boundscheck checkbounds(x, I...)
    @inbounds begin
        index = x.parent[I...]
        nzindex = x.nzinds[index]
    end
    @boundscheck @assert nzindex != -1
    NonzeroIndex(index, nzindex)
end
@inline nonzeroindices(space::MPSpace, inds) = (@_propagate_inbounds_meta; _nonzeroindices(get_gridsppat(space), inds))
_nonzeroindices(::Trues, inds) = inds
@inline function _nonzeroindices(sppat::SpPattern, inds)
    @boundscheck checkbounds(sppat, inds)
    NonzeroIndices(inds, get_spindices(sppat))
end

function update!(space::MPSpace{dim, T}, grid::Grid, particles::Particles; filter::Union{Nothing, AbstractArray{Bool}} = nothing) where {dim, T}
    @assert num_particles(space) == length(particles)

    update!(get_particlesinblocks(space), get_lattice(grid), particles.x)
    #
    # Following `update_mpvalues!` update `space.sppat` and use it when `filter` is given.
    # This consideration of sparsity pattern is necessary in some `Interpolation`s such as `WLS` and `KernelCorrection`.
    # However, this updated sparsity pattern is not used for updating sparsity pattern of grid-state because
    # the inactive nodes also need their values (even zero) for `NonzeroIndex` used in P2G.
    # Thus, `update_sparsity_pattern!` must be executed after `update_mpvalues!`.
    #
    #            |      |      |                             |      |      |
    #         ---×------×------×---                       ---●------●------●---
    #            |      |      |                             |      |      |
    #            |      |      |                             |      |      |
    #         ---×------●------●---                       ---●------●------●---
    #            |      |      |                             |      |      |
    #            |      |      |                             |      |      |
    #         ---●------●------●---                       ---●------●------●---
    #            |      |      |                             |      |      |
    #
    #   < Sparsity pattern for `MPValue` >     < Sparsity pattern for Grid-state (`SpArray`) >
    #
    update_mpvalues!(space, get_lattice(grid), particles, filter)
    update_sparsity_pattern!(space)
    unsafe_update_sparsity_pattern!(grid, get_sppat(space))

    space
end

# block-wise rough sparsity pattern for grid-state
# Don't use "exact" sparsity pattern because it requires iteraion over all particles
# CubicBSpline is still ok with rng=1
function update_sparsity_pattern!(space::MPSpace, rng::Int = 1)
    sppat = fillzero!(get_sppat(space))
    ptsinblks = get_particlesinblocks(space)
    @inbounds for I in CartesianIndices(ptsinblks)
        blk = ptsinblks[I]
        if !isempty(blk)
            inds = neighbornodes_from_blockindex(gridsize(space), I, rng)
            sppat[inds] .= true
        end
    end
end
@inline function neighbornodes_from_blockindex(gridsize::Dims, blk::CartesianIndex, i::Int)
    start = @. max((blk.I-1) << BLOCK_UNIT + 1 - i, 1)
    stop = @. min(blk.I << BLOCK_UNIT + 1 + i, gridsize)
    CartesianIndex(start):CartesianIndex(stop)
end

function update_mpvalues!(space::MPSpace, lattice::Lattice, particles::Particles, filter::Union{Nothing, AbstractArray{Bool}})
    @assert gridsize(space) == size(lattice)
    @assert length(particles) == num_particles(space)

    if filter === nothing
        update!(values(space), get_interpolation(space), lattice, particles)
    else
        # handle excluded domain
        sppat = get_sppat(space)
        sppat .= filter
        parallel_each_particle(space) do p
            @inbounds begin
                inds = neighbornodes(lattice, particles.x[p], 1)
                sppat[inds] .= true
            end
        end
        update!(values(space), get_interpolation(space), lattice, sppat, particles)
    end

    space
end

# block-wise parallel computation
function parallel_each_particle(f, ptsinblks::AbstractArray)
    for blocks in threadsafe_blocks(size(ptsinblks))
        ptsinblks′ = filter(!isempty, view(ptsinblks, blocks))
        @threaded_inbounds for pinds in ptsinblks′
            foreach(f, pinds)
        end
    end
end
function parallel_each_particle(f, space::MPSpace)
    parallel_each_particle(f, get_particlesinblocks(space))
end
