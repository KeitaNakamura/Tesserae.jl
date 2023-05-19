"""
    MPSpace(interpolation, gridsize, num_particles)

Create `interpolation` space.
"""
struct MPSpace{dim, T, It <: Interpolation, V, VI, BS <: BlockSpace{dim}}
    interp::It
    mpvals::MPValues{dim, T, V, VI}
    blkspace::BS
    gridspinds::Base.RefValue{Any}
    tmp_sppat::Array{Bool, dim} # used for filtering case
    tmp_sppat_blk::Array{Bool, dim}
end

# constructors
function MPSpace(::Type{T}, itp::Interpolation, gridsize::Dims{dim}, npts::Integer) where {dim, T}
    blksize = blocksize(gridsize)
    mpvals = MPValues{dim, T}(itp, npts)
    blkspace = BlockSpace(blksize, npts)
    tmp_sppat = fill(false, gridsize)
    tmp_sppat_blk = fill(false, blksize)
    MPSpace(itp, mpvals, blkspace, Ref{Any}(), tmp_sppat, tmp_sppat_blk)
end
MPSpace(itp::Interpolation, gridsize::Dims, npts::Integer) = MPSpace(Float64, itp, gridsize, npts)

# helper functions
gridsize(space::MPSpace) = size(space.tmp_sppat)
num_particles(space::MPSpace) = num_particles(values(space))
get_interpolation(space::MPSpace) = space.interp
get_blockspace(space::MPSpace) = space.blkspace
get_gridspinds(space::MPSpace) = space.gridspinds[]
set_gridspinds!(space::MPSpace, spinds) = space.gridspinds[] = spinds

reorder_particles!(particles::Particles, space::MPSpace) = reorder_particles!(particles, get_blockspace(space))

# values
Base.values(space::MPSpace) = space.mpvals
Base.values(space::MPSpace, i::Integer) = (@_propagate_inbounds_meta; values(space.mpvals, i))

"""
    update!(space::MPSpace, grid, particles)

Update interpolation `space`.

This must be done before calling [`particle_to_grid!`](@ref) and [`grid_to_particle!`](@ref).

!!! note "Update of sparsity pattern of `SpArray` in `grid`"
    If `grid` has [`SpArray`](@ref)s, their sparsity pattern is also updated based on the position of `particles`.
    This updated sparsity pattern is block-wise rough pattern rather than precise pattern for the performance.
    See also "[How to use grid](@ref)".
"""
function update!(space::MPSpace, grid::Grid, particles::Particles; filter::Union{Nothing, AbstractArray{Bool}}=nothing, parallel::Bool=true)
    @assert gridsize(space) == size(grid)
    @assert num_particles(space) == length(particles)

    update!(get_blockspace(space), get_lattice(grid), particles.x; parallel)
    update_mpvalues!(space, get_lattice(grid), particles, filter; parallel)

    if grid isa SpGrid
        sppat_blk = fillzero!(space.tmp_sppat_blk)
        update_sparsity_pattern!(sppat_blk, get_blockspace(space))
        update_sparsity_pattern!(grid, sppat_blk)
        set_gridspinds!(space, get_spinds(grid))
    else
        set_gridspinds!(space, nothing)
    end

    space
end

function update_mpvalues!(space::MPSpace, lattice::Lattice, particles::Particles, filter::Union{Nothing, AbstractArray{Bool}}; parallel::Bool)
    @assert gridsize(space) == size(lattice)
    @assert length(particles) == num_particles(space)

    if filter === nothing
        update!(values(space), get_interpolation(space), lattice, particles; parallel)
    else
        # handle excluded domain
        sppat = space.tmp_sppat
        sppat .= filter
        parallel_each_particle(space; parallel) do p
            @inbounds begin
                inds = neighbornodes(lattice, particles.x[p], 1)
                sppat[inds] .= true
            end
        end
        update!(values(space), get_interpolation(space), lattice, sppat, particles; parallel)
    end

    space
end

function parallel_each_particle(f, space::MPSpace; parallel::Bool=true)
    parallel_each_particle(f, get_blockspace(space), num_particles(space); parallel)
end
