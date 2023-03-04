struct MPSpace{dim, T, It <: Interpolation, V, VI, Gs <: Union{Trues, SpPattern}}
    interp::It
    mpvals::MPValues{dim, T, V, VI}
    ptspblk::Array{PushVector{Int}, dim}
    sppat::Array{Bool, dim}
    gridsppat::Gs # sppat used in SpGrid
end

# constructors
function MPSpace(itp::Interpolation, lattice::Lattice{dim, T}, xₚ::AbstractVector{<: Vec{dim}}, gridsppat) where {dim, T}
    npts = length(xₚ)
    mpvals = MPValues{dim, T}(itp, npts)
    sppat = fill(false, size(lattice))
    MPSpace(itp, mpvals, pointsperblock(lattice, xₚ), sppat, gridsppat)
end
MPSpace(itp::Interpolation, grid::Grid, particles::Particles) = MPSpace(itp, get_lattice(grid), particles.x, get_sppat(grid))

# helper functions
gridsize(space::MPSpace) = size(space.sppat)
num_particles(space::MPSpace) = length(space.mpvals)
get_interpolation(space::MPSpace) = space.interp
get_pointsperblock(space::MPSpace) = space.ptspblk
get_sppat(space::MPSpace) = space.sppat
get_gridsppat(space::MPSpace) = space.gridsppat

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

# reorder_particles!
function reorder_particles!(particles::Particles, ptspblk::Array)
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

    # keep missing particles aside
    if length(inds) != length(particles) # some points are missing
        missed = particles[setdiff(1:length(particles), inds)]
    end

    # reorder particles
    @inbounds particles[1:length(inds)] .= view(particles, inds)

    # assign missing particles to the end part of `particles`
    if length(inds) != length(particles)
        @inbounds particles[length(inds)+1:end] .= missed
    end

    particles
end
reorder_particles!(particles::Particles, space::MPSpace) = reorder_particles!(particles, get_pointsperblock(space))

# pointsperblock!
function pointsperblock!(ptspblk::AbstractArray, lattice::Lattice, xₚ::AbstractVector)
    empty!.(ptspblk)
    @inbounds for p in 1:length(xₚ)
        I = whichblock(lattice, xₚ[p])
        I === nothing || push!(ptspblk[I], p)
    end
    finish!.(ptspblk)
    ptspblk
end
function pointsperblock(lattice::Lattice, xₚ::AbstractVector)
    ptspblk = Array{PushVector{Int}}(undef, blocksize(size(lattice)))
    @inbounds for i in eachindex(ptspblk)
        ptspblk[i] = PushVector{Int}()
    end
    pointsperblock!(ptspblk, lattice, xₚ)
end
function update_pointsperblock!(space::MPSpace, lattice::Lattice, xₚ::AbstractVector)
    pointsperblock!(get_pointsperblock(space), lattice, xₚ)
end

function update!(space::MPSpace{dim, T}, grid::Grid, particles::Particles; filter::Union{Nothing, AbstractArray{Bool}} = nothing) where {dim, T}
    @assert num_particles(space) == length(particles)

    update_pointsperblock!(space, get_lattice(grid), particles.x)
    #
    # Following `update_mpvalues_neighbornodes!` update `space.sppat` and use it when `filter` is given.
    # This consideration of sparsity pattern is necessary in some `Interpolation`s such as `WLS` and `KernelCorrection`.
    # However, this updated sparsity pattern is not used for updating sparsity pattern of grid-state because
    # the inactive nodes also need their values (even zero) for `NonzeroIndex` used in P2G.
    # Thus, `update_sparsity_pattern!` must be executed after `update_mpvalues_neighbornodes!`.
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
    update_mpvalues_neighbornodes!(space, get_lattice(grid), particles, filter)
    update_sparsity_pattern!(space)
    update_sparsity_pattern!(grid, space)

    space
end

# block-wise rough sparsity pattern for grid-state
# Don't use "exact" sparsity pattern because it requires iteraion over all particles
# CubicBSpline is still ok with rng=1
function update_sparsity_pattern!(space::MPSpace, rng::Int = 1)
    sppat = fillzero!(get_sppat(space))
    ptspblk = get_pointsperblock(space)
    @inbounds for I in CartesianIndices(ptspblk)
        blk = ptspblk[I]
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

function update_mpvalues_neighbornodes!(space::MPSpace, lattice::Lattice, particles::Particles, filter::Union{Nothing, AbstractArray{Bool}})
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

function update_sparsity_pattern!(grid::SpGrid, space::MPSpace)
    @assert size(grid) == gridsize(space)
    unsafe_update_sparsity_pattern!(grid, get_sppat(space))
    grid
end
update_sparsity_pattern!(grid::Grid, space::MPSpace) = grid

# block-wise parallel computation
function parallel_each_particle(f, ptspblk::Array)
    for blocks in threadsafe_blocks(size(ptspblk))
        ptspblk′ = filter(!isempty, view(ptspblk, blocks))
        @threaded_inbounds for pinds in ptspblk′
            foreach(f, pinds)
        end
    end
end
function parallel_each_particle(f, space::MPSpace)
    parallel_each_particle(f, get_pointsperblock(space))
end
