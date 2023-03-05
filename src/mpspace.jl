struct MPSpace{dim, T, It <: Interpolation, V, VI, B <: ParticlesInBlocks, GS <: Union{Trues, SpPattern}}
    interp::It
    mpvals::MPValues{dim, T, V, VI}
    ptsinblks::B
    sppat::Array{Bool, dim}
    gridsppat::GS # sppat used in SpGrid
end

# constructors
function MPSpace(itp::Interpolation, lattice::Lattice{dim, T}, xₚ::AbstractVector{<: Vec{dim}}, gridsppat) where {dim, T}
    npts = length(xₚ)
    mpvals = MPValues{dim, T}(itp, npts)
    ptsinblks = ParticlesInBlocks(blocksize(lattice), npts)
    sppat = fill(false, size(lattice))
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

    update_sparsity_pattern!(get_particlesinblocks(space), get_lattice(grid), particles.x)
    #
    # Following `update_mpvalues!` updates `space.sppat` and use it when `filter` is given.
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
    update_sparsity_pattern!(get_sppat(space), get_particlesinblocks(space))
    unsafe_update_sparsity_pattern!(grid, get_sppat(space))

    space
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
