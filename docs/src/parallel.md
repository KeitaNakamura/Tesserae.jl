# Parallel execution

Multi-threading in Tesserae parallelizes CPU work over particles, grid nodes, and particle-grid transfers.
The [`@threaded`](@ref) macro adds this parallelism while keeping transfer expressions close to their sequential form.
It behaves similarly to Julia's built-in `Threads.@threads`, but is designed to work with particle-grid transfer macros such as [`@G2P`](@ref), [`@P2G`](@ref), [`@G2P2G`](@ref), and [`@P2G_Matrix`](@ref).

## Usage guidelines

Particle-grid transfers have two directions: gathering and scattering.
In a gathering transfer, each particle reads values from nearby grid nodes, so the operation can be threaded directly.
In a scattering transfer, particles write contributions to grid nodes.
If multiple threads update the same grid node at the same time, this is a data race; see Julia's discussion of [data races between threads](https://docs.julialang.org/en/v1/manual/multi-threading/#Communication-and-data-races-between-threads).
Threaded scattering therefore uses a [`Partition`](@ref).

### Gathering (`@G2P`)

To parallelize `@G2P`, simply prefix it with `@threaded`.

```julia
@threaded @G2P grid=>i particles=>p weights=>ip begin
    # your code here
end
```

### Scattering (`@P2G`, `@G2P2G` and `@P2G_Matrix`)

For scattering operations, prefix `@P2G` with `@threaded` and use [`Partition`](@ref) to avoid data races on the grid.

```julia
partition = Partition(mesh)
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

For `@P2G` and related scattering operations, using `reorder_particles!` together with `Partition` can significantly improve cache efficiency and thread scaling:

```julia
partition = Partition(mesh)
update!(partition, particles.x)
reorder_particles!(particles, partition)
```

Reordering ensures that particles within the same grid block are stored contiguously in memory, reducing random memory access during parallel execution.

In a step loop, call it every step with a `threshold` below `1`, and let it decide which steps to act on. It permutes `particles` and nothing else, so it belongs before the basis weights are computed for them:

```julia
update!(partition, particles.x)
reorder_particles!(particles, partition; threshold=0.85)
update!(weights, particles, grid.x)
```

For `0 ≤ threshold ≤ 1`, larger values reorder more often. Particles are reordered when [`Tesserae.block_ordered_particle_contiguity`](@ref) is below `threshold`. At the endpoints, `threshold=0` never reorders and `threshold=1`, the default, reorders on every call. `reorder_particles!` returns `true` on the calls where it reordered.

!!! note
    Reordering on every step makes each transfer as fast as it can be and is usually still slower overall, because reordering moves about as many bytes as the transfer it speeds up.
    How far the order drifts per step depends on how fast particles cross blocks, so the threshold worth using varies with the problem.


## Multi-threading API

```@docs
@threaded
Partition
reorder_particles!
Tesserae.block_ordered_particle_contiguity
```
