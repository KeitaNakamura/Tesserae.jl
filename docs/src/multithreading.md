# Multi-threading

Tesserae.jl provides a [`@threaded`](@ref) macro to enable multi-threading. It behaves similarly to Julia’s built-in Threads.@threads, but is specifically designed to work with particle-grid transfer macros such as [`@G2P`](@ref), [`@P2G`](@ref), [`@G2P2G`](@ref), and [`@P2G_Matrix`](@ref).

## Usage guidelines

### Gathering (`@G2P`)

To parallelize `@G2P`, simply prefix it with `@threaded`.

```julia
@threaded @G2P grid=>i particles=>p weights=>ip begin
    # your code here
end
```

### Scattering (`@P2G`, `@G2P2G` and `@P2G_Matrix`)

For scattering operations, prefix `@P2G` with `@threaded` and use [`ColorPartition`](@ref) to avoid data races on the grid.

```julia
partition = ColorPartition(mesh)
update!(partition, particles.x)
@threaded @P2G grid=>i particles=>p weights=>ip partition begin
    # your code here
end
```

Same applies to `@G2P2G` and `@P2G_Matrix`.

### Updating interpolation values

To update interpolation values, either use the [`update!`](@ref) function, or simply:

```julia
@threaded for p in eachindex(particles)
    update!(weights[p], particles.x[p], mesh)
end
```

### Reordering particles

For `@P2G` and related scattering operations, using `reorder_particles!` together with `ColorPartition` can significantly improve cache efficiency and thread scaling:

```julia
partition = ColorPartition(mesh)
update!(partition, particles.x)
reorder_particles!(particles, partition)
```

Reordering ensures that particles within the same grid block are stored contiguously in memory, reducing random memory access during parallel execution.

!!! warning
    `reorder_particles!` can be expensive for large systems.
    It is usually sufficient to reorder particles only when their spatial distribution has changed significantly.
    Avoid calling it on every step.


## Multi-threading API

```@docs
@threaded
ColorPartition
```
