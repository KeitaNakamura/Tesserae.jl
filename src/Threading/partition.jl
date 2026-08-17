# -----------------------------------------------------------------------------
#  ThreadPartition
# -----------------------------------------------------------------------------

# ---- ThreadPartition ----

"""
    ThreadPartition(::CartesianMesh)
    ThreadPartition(::FEMesh)
    ThreadPartition(::IGAMesh)

`ThreadPartition` stores partitioning information used by the [`@P2G`](@ref), [`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) macros
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
# Construct ThreadPartition
partition = ThreadPartition(mesh)

# Update partition using current particle positions
update!(partition, particles.x) # Required only for `CartesianMesh`.

# P2G transfer
@threaded @P2G grid=>i particles=>p weights=>ip partition begin
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
end
```
"""
struct ThreadPartition{Strategy <: PartitionStrategy}
    strategy::Strategy
end

strategy(partition::ThreadPartition) = partition.strategy
threadsafe_groups(partition::ThreadPartition) = threadsafe_groups(strategy(partition))

particle_indices(partition::ThreadPartition, particles, region) =
    particle_indices(strategy(partition), region)
particle_indices(partition::ThreadPartition{<: CellStrategy}, particles, cell) =
    (CartesianIndex(p, cell) for p in 1:size(particles, 1))

ThreadPartition(mesh::CartesianMesh) = ThreadPartition(BlockStrategy(mesh))
ThreadPartition(mesh::AbstractCellMesh) = ThreadPartition(CellStrategy(mesh))
update!(partition::ThreadPartition, args...) = update!(strategy(partition), args...)

reorder_particles!(particles::StructVector, partition::ThreadPartition{<: BlockStrategy}; kwargs...) =
    reorder_particles!(particles, strategy(partition); kwargs...)
block_ordered_particle_contiguity(partition::ThreadPartition{<: BlockStrategy}) =
    block_ordered_particle_contiguity(strategy(partition))
reorder_particles!(particles, ::ThreadPartition{<: GPUBlockStrategy}; kwargs...) =
    error("reorder_particles! does not support GPU partitions yet")
block_ordered_particle_contiguity(::ThreadPartition{<: GPUBlockStrategy}) =
    error("block_ordered_particle_contiguity does not support GPU partitions yet")
