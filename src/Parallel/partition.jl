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
    (CartesianIndex(p, cell) for p in 1:size(particles, 1))

Partition(mesh::CartesianMesh) = Partition(CPUBlockStrategy(mesh))
Partition(mesh::AbstractCellMesh) = Partition(CellStrategy(mesh))
update!(partition::Partition, args...) = update!(strategy(partition), args...)

reorder_particles!(particles::StructVector, partition::Partition{<: Union{CPUBlockStrategy, GPUBlockStrategy}}; kwargs...) =
    reorder_particles!(particles, strategy(partition); kwargs...)
block_ordered_particle_contiguity(partition::Partition{<: Union{CPUBlockStrategy, GPUBlockStrategy}}) =
    block_ordered_particle_contiguity(strategy(partition))
