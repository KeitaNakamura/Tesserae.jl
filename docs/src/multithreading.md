# Multi-threading

Multi-threading in Tesserae parallelizes CPU work over particles, grid nodes, and particle-grid transfers.
The [`@threaded`](@ref) macro adds this parallelism while keeping transfer expressions close to their sequential form.
It behaves similarly to Julia's built-in `Threads.@threads`, but is designed to work with particle-grid transfer macros such as [`@G2P`](@ref), [`@P2G`](@ref), [`@G2P2G`](@ref), and [`@P2G_Matrix`](@ref).

## Usage guidelines

Particle-grid transfers have two directions: gathering and scattering.
In a gathering transfer, each particle reads values from nearby grid nodes, so the operation can be threaded directly.
In a scattering transfer, particles write contributions to grid nodes.
If multiple threads update the same grid node at the same time, this is a data race; see Julia's discussion of [data races between threads](https://docs.julialang.org/en/v1/manual/multi-threading/#Communication-and-data-races-between-threads).
Threaded scattering therefore uses a [`ThreadPartition`](@ref).

### Gathering (`@G2P`)

To parallelize `@G2P`, simply prefix it with `@threaded`.

```julia
@threaded @G2P grid=>i particles=>p weights=>ip begin
    # your code here
end
```

### Scattering (`@P2G`, `@G2P2G` and `@P2G_Matrix`)

For scattering operations, prefix `@P2G` with `@threaded` and use [`ThreadPartition`](@ref) to avoid data races on the grid.

```julia
partition = ThreadPartition(mesh)
update!(partition, particles.x) # CartesianMesh only
@threaded @P2G grid=>i particles=>p weights=>ip partition begin
    # your code here
end
```

For [`FEMesh`](@ref) and [`IGAMesh`](@ref), the partition is built from the
fixed cell connectivity, so it does not need an `update!` call. The same
partitioning applies to `@G2P2G` and `@P2G_Matrix`.

### Updating basis weights

To update basis weights, either use the [`update!`](@ref) function, or simply:

```julia
@threaded for p in eachindex(particles)
    update!(weights[p], particles.x[p], mesh)
end
```

### Reordering particles

For `@P2G` and related scattering operations, using `reorder_particles!` together with `ThreadPartition` can significantly improve cache efficiency and thread scaling:

```julia
partition = ThreadPartition(mesh)
update!(partition, particles.x)
reorder_particles!(particles, partition)
```

Reordering ensures that particles within the same grid block are stored contiguously in memory, reducing random memory access during parallel execution.
When reordering is checked every step, use an adaptive threshold such as `threshold=0.85`:

```julia
update!(partition, particles.x)
reorder_particles!(particles, partition; threshold=0.85)
```

For `0 ≤ threshold ≤ 1`, larger values reorder more often. Particles are reordered when [`Tesserae.block_ordered_particle_contiguity`](@ref) is below `threshold`.

At the endpoints, `threshold=0` never reorders and `threshold=1` always reorders.

!!! warning
    `reorder_particles!` can be expensive for large systems.
    It is usually sufficient to reorder particles only when their spatial distribution has changed significantly.
    Avoid forcing it on every step unless the P2G speedup is worth the reorder cost.


## Multi-threading API

```@docs
@threaded
ThreadPartition
reorder_particles!
Tesserae.block_ordered_particle_contiguity
```
