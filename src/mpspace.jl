struct MPSpace{dim, T, I <: Interpolation, MP <: MPValue{dim, T, I}}
    sppat::Array{Bool, dim}
    mpvals::Vector{MP}
    nodeinds::Vector{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}}
    ptspblk::Array{Vector{Int}, dim}
    stamp::RefValue{Float64}
end

# constructors
function MPSpace(interp::Interpolation, lattice::Lattice{dim, T}, xₚ::AbstractVector{<: Vec}) where {dim, T}
    sppat = fill(false, size(lattice))
    npts = length(xₚ)
    mpvals = [MPValue{dim, T}(interp) for _ in 1:npts]
    nodeinds = [CartesianIndices(nfill(1:0, Val(dim))) for _ in 1:npts]
    MPSpace(sppat, mpvals, nodeinds, pointsperblock(lattice, xₚ), Ref(NaN))
end
MPSpace(interp::Interpolation, grid::Grid, particles::StructVector) = MPSpace(interp, get_lattice(grid), particles.x)

# helper functions
gridsize(space::MPSpace) = size(space.sppat)
num_particles(space::MPSpace) = length(space.mpvals)
get_mpvalue(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.mpvals[i])
get_nodeindices(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.nodeinds[i])
set_nodeindices!(space::MPSpace, i::Int, x) = (@_propagate_inbounds_meta; space.nodeinds[i] = x)
get_sppat(space::MPSpace) = space.sppat
get_pointsperblock(space::MPSpace) = space.ptspblk
get_stamp(space::MPSpace) = space.stamp[]

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
    rest = particles[setdiff(1:length(particles), inds)]
    @inbounds begin
        particles[1:length(inds)] .= view(particles, inds)
        particles[length(inds)+1:end] .= rest
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

    space.stamp[] = time()
    update_pointsperblock!(space, get_lattice(grid), particles.x)
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

function update_mpvalues!(space::MPSpace, lattice::Lattice, particles::Particles, filter::Union{Nothing, AbstractArray{Bool}})
    # following fields are updated in this function
    # * `space.mpvals`
    # * `space.nodeinds`

    @assert length(particles) == num_particles(space)

    @threaded for p in 1:num_particles(space)
        mp = get_mpvalue(space, p)
        pt = LazyRow(particles, p)
        set_nodeindices!(space, p, neighbornodes(mp, lattice, pt))
        # normally update mpvalues here
        filter===nothing && update!(get_mpvalue(space, p), lattice, AllTrue(), get_nodeindices(space, p), pt)
    end

    # handle excluded domain
    if filter !== nothing
        sppat = get_sppat(space)
        sppat .= filter
        parallel_each_particle(space) do p
            @inbounds begin
                inds = neighbornodes(lattice, particles.x[p], 1)
                sppat[inds] .= true
            end
        end
        # update mpvalues after completely updating `sppat`
        # but this `sppat` is not used for `SpArray` for grid
        @threaded for p in 1:num_particles(space)
            update!(get_mpvalue(space, p), lattice, sppat, get_nodeindices(space, p), LazyRow(particles, p))
        end
    end

    space
end

function update_sparsity_pattern!(grid::SpGrid, space::MPSpace)
    @assert size(grid) == gridsize(space)
    update_sparsity_pattern!(grid, get_sppat(space))
    set_stamp!(grid, get_stamp(space))
    grid
end
update_sparsity_pattern!(grid::Grid, space::MPSpace) = grid

# block-wise parallel computation
function parallel_each_particle(f, space::MPSpace)
    for blocks in threadsafe_blocks(blocksize(gridsize(space)))
        pointsperblock = collect(Iterators.filter(!isempty, Iterators.map(blkidx->get_pointsperblock(space)[blkidx], blocks)))
        @threaded for pinds in pointsperblock
            foreach(f, pinds)
        end
    end
end
