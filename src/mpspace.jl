struct MPSpace{dim, T, MP <: MPValue{dim, T}, GS <: Union{Nothing, SpPattern}}
    sppat::Array{Bool, dim}
    mpvals::Vector{MP}
    nodeinds::Vector{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}}
    ptspblk::Array{Vector{Int}, dim}
    gridsppat::GS # sppat used in SpGrid
end

# constructors
function MPSpace(itp::Interpolation, lattice::Lattice{dim, T}, xₚ::AbstractVector{<: Vec{dim}}, gridsppat) where {dim, T}
    sppat = fill(false, size(lattice))
    npts = length(xₚ)
    mpvals = [MPValue{dim, T}(itp) for _ in 1:npts]
    nodeinds = [CartesianIndices(nfill(1:0, Val(dim))) for _ in 1:npts]
    MPSpace(sppat, mpvals, nodeinds, pointsperblock(lattice, xₚ), gridsppat)
end
MPSpace(itp::Interpolation, grid::Grid, particles::Particles) = MPSpace(itp, get_lattice(grid), particles.x, nothing)
MPSpace(itp::Interpolation, grid::SpGrid, particles::Particles) = MPSpace(itp, get_lattice(grid), particles.x, get_sppat(grid))

# helper functions
gridsize(space::MPSpace) = size(space.sppat)
num_particles(space::MPSpace) = length(space.mpvals)
get_sppat(space::MPSpace) = space.sppat
get_gridsppat(space::MPSpace) = space.gridsppat
get_pointsperblock(space::MPSpace) = space.ptspblk

# mpvalue
mpvalue(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.mpvals[i])
# set/get gridindices
@inline neighbornodes(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; _neighbornodes(space.gridsppat, space, i))
@inline _neighbornodes(::Nothing, space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.nodeinds[i])
@inline _neighbornodes(sppat::SpPattern, space::MPSpace, i::Int) =
    (@_propagate_inbounds_meta; Iterators.map(I -> NonzeroIndex(I, @inbounds(sppat.indices[I])), space.nodeinds[i]))
set_gridindices!(space::MPSpace, i::Int, x) = (@_propagate_inbounds_meta; space.nodeinds[i] = x)

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
function pointsperblock!(ptspblk::AbstractArray{Vector{Int}}, lattice::Lattice, xₚ::AbstractVector)
    empty!.(ptspblk)
    @inbounds for p in 1:length(xₚ)
        I = whichblock(lattice, xₚ[p])
        I === nothing || push!(ptspblk[I], p)
    end
    ptspblk
end
function pointsperblock(lattice::Lattice, xₚ::AbstractVector)
    ptspblk = Array{Vector{Int}}(undef, blocksize(size(lattice)))
    @inbounds for i in eachindex(ptspblk)
        ptspblk[i] = Int[]
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
@inline function neighbornodes_from_blockindex(gridsize::Dims, blk::CartesianIndex, i::Int)
    lo = blk.I .- i
    hi = blk.I .+ i
    start = @. max((lo-1) << BLOCK_UNIT + 1, 1)
    stop = @. min(hi << BLOCK_UNIT + 1, gridsize)
    CartesianIndex(start):CartesianIndex(stop)
end

function update_mpvalues_neighbornodes!(space::MPSpace, lattice::Lattice, particles::Particles, filter::Union{Nothing, AbstractArray{Bool}})
    @assert gridsize(space) == size(lattice)
    @assert length(particles) == num_particles(space)

    if filter === nothing
        update_mpvalues_neighbornodes!(space, lattice, Trues(size(lattice)), particles)
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
        update_mpvalues_neighbornodes!(space, lattice, sppat, particles)
    end

    space
end
function update_mpvalues_neighbornodes!(space::MPSpace, lattice::Lattice, sppat::AbstractArray{Bool}, particles::Particles)
    @threaded for p in 1:num_particles(space)
        indices = update!(mpvalue(space, p), lattice, sppat, LazyRow(particles, p))
        set_gridindices!(space, p, indices)
    end
end

function update_sparsity_pattern!(grid::SpGrid, space::MPSpace)
    @assert size(grid) == gridsize(space)
    unsafe_update_sparsity_pattern!(grid, get_sppat(space))
    grid
end
update_sparsity_pattern!(grid::Grid, space::MPSpace) = grid

# block-wise parallel computation
function parallel_each_particle(f, ptspblk::Array{Vector{Int}})
    for blocks in threadsafe_blocks(size(ptspblk))
        ptspblk′ = collect(Iterators.filter(!isempty, view(ptspblk, blocks)))
        @threaded for pinds in ptspblk′
            foreach(f, pinds)
        end
    end
end
function parallel_each_particle(f, space::MPSpace)
    parallel_each_particle(f, get_pointsperblock(space))
end
