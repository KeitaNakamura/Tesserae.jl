# -----------------------------------------------------------------------------
#  Partition
# -----------------------------------------------------------------------------

"""
    Partition(::CartesianMesh)
    Partition(::FEMesh)
    Partition(::IGAMesh)

`Partition` stores partitioning information used by the [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) macros
to avoid write conflicts during threaded particle-to-grid transfers.

On GPU the same type schedules workgroups instead of threads: `gpu(partition)`
rebuilds it for the device, and passing it to [`@P2G`](@ref) selects a
block-scheduled kernel that accumulates each grid block in shared memory. There
`@threaded` plays no part, and [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) do not
take a device partition yet.

!!! note
    The [`@threaded`](@ref) macro must be placed before [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) to enable parallel transfer on CPU.

# Examples
```julia
# Construct Partition
partition = Partition(mesh)

# Update partition using current particle positions
update!(partition, particles.x) # Required only for `CartesianMesh`.

# P2G transfer
@threaded @P2G grid=>i particles=>p weights=>ip partition begin
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
end
```
"""
struct Partition{Strategy <: PartitionStrategy}
    strategy::Strategy
end

strategy(partition::Partition) = partition.strategy
threadsafe_groups(partition::Partition) = threadsafe_groups(strategy(partition))

particle_indices(partition::Partition, particles, region) =
    particle_indices(strategy(partition), region)
particle_indices(partition::Partition{<: CellStrategy}, particles, cell) =
    cell_quadrature_indices(particles, cell)

Partition(mesh::CartesianMesh) = Partition(CPUBlockStrategy(mesh))
Partition(mesh::AbstractCellMesh) = Partition(CellStrategy(mesh))
update!(partition::Partition, args...) = update!(strategy(partition), args...)

"""
    reorder_particles!(particles, partition; threshold=1)

Reorder particles by the current block partition, and return whether it did.

Particles are reordered when [`Tesserae.block_ordered_particle_contiguity`](@ref)
is below `threshold`, which by default is every call. For `0 ≤ threshold ≤ 1`,
larger values reorder more often; `threshold=0` never reorders.

In a step loop, call this every step but pass a `threshold` below `1`, such as
`0.85`, and let it decide which steps to act on. Reordering moves about as many
bytes as the transfer it speeds up, so reordering on every step usually costs
more than it saves.

On a partition moved with `gpu`, the reorder runs on the device through the
partition's block-sorted permutation. Unlike the CPU path, particles outside
the mesh are an error there rather than being kept at the end of the array.

!!! warning
    This permutes `particles` and nothing else, so anything already computed per
    particle -- basis weights above all -- is stale afterwards. Call it before
    `update!(weights, particles, mesh)`, not between that and the transfer.
"""
reorder_particles!(particles::StructVector, partition::Partition{<: BlockStrategy}; kwargs...) =
    reorder_particles!(particles, strategy(partition); kwargs...)

function reorder_particles!(particles::StructVector, bs::BlockStrategy; threshold=1)
    0 ≤ threshold ≤ 1 || throw(ArgumentError("threshold must be in [0, 1]."))
    iszero(threshold) && return false
    if threshold == 1 || block_ordered_particle_contiguity(bs) < threshold
        _reorder_partition_particles!(particles, bs)
        return true
    end
    return false
end

block_ordered_particle_contiguity(partition::Partition{<: BlockStrategy}) =
    block_ordered_particle_contiguity(strategy(partition))
