struct MPSpace{dim, T, It <: Interpolation, V, VI}
    interp::It
    mpvals::MPValues{dim, T, V, VI}
    blkspace::BlockSpace{dim}
    sppat::Array{Bool, dim}
    gridsppat::Base.RefValue{Any}
end

# constructors
function MPSpace{dim, T}(itp::Interpolation, gridsize::Dims, npts::Integer) where {dim, T}
    mpvals = MPValues{dim, T}(itp, npts)
    blkspace = BlockSpace(blocksize(gridsize), npts)
    sppat = fill(false, gridsize)
    MPSpace(itp, mpvals, blkspace, sppat, Ref{Any}())
end
MPSpace(itp::Interpolation, lattice::Lattice{dim, T}, xₚ::AbstractVector) where {dim, T} = MPSpace{dim, T}(itp, size(lattice), length(xₚ))
MPSpace(itp::Interpolation, grid::Grid, particles::Particles) = MPSpace(itp, get_lattice(grid), particles.x)

# helper functions
gridsize(space::MPSpace) = size(space.sppat)
num_particles(space::MPSpace) = length(space.mpvals)
get_interpolation(space::MPSpace) = space.interp
get_blockspace(space::MPSpace) = space.blkspace
get_sppat(space::MPSpace) = space.sppat
get_gridsppat(space::MPSpace) = space.gridsppat[]
set_gridsppat!(space::MPSpace, sppat) = space.gridsppat[] = sppat

reorder_particles!(particles::Particles, space::MPSpace) = reorder_particles!(particles, get_blockspace(space))

# values
Base.values(space::MPSpace) = space.mpvals
Base.values(space::MPSpace, i::Integer) = (@_propagate_inbounds_meta; values(space.mpvals, i))
# neighbornodes
@inline neighbornodes(space::MPSpace, i::Integer) = (@_propagate_inbounds_meta; neighbornodes(values(space), i))
@inline neighbornodes(space::MPSpace, ::Grid, i::Integer) = (@_propagate_inbounds_meta; neighbornodes(space, i))
@inline neighbornodes(space::MPSpace, grid::SpGrid, i::Integer) = (@_propagate_inbounds_meta; nonzeroindices(get_sppat(grid), neighbornodes(space, i)))

function update!(space::MPSpace{dim, T}, grid::Grid, particles::Particles; filter::Union{Nothing, AbstractArray{Bool}} = nothing) where {dim, T}
    @assert num_particles(space) == length(particles)

    update!(get_blockspace(space), get_lattice(grid), particles.x)
    #
    # When `filter` is given, following `update_mpvalues!` updates `space.sppat`, too.
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
    update_sparsity_pattern!(get_sppat(space), get_blockspace(space))
    unsafe_update_sparsity_pattern!(grid, get_sppat(space))
    set_gridsppat!(space, get_sppat(grid))

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

function parallel_each_particle(f, space::MPSpace)
    parallel_each_particle(f, get_blockspace(space))
end
