# SpArray

`SpArray` is a block-sparse array for grid fields on a [`CartesianMesh`](@ref).
It has the same logical size as a dense array, but allocates storage only for active blocks.

The main use case is a large Cartesian mesh where the simulated material occupies only a small part of the domain.
At this stage, `SpArray` is mainly a way to avoid allocating grid fields over empty regions.
It should not be expected to make every computation much faster, but it can substantially reduce memory use when a dense grid would contain many nodes that are never touched by particles.

## Sparse grid

For grid-based computations, create a sparse grid by passing `SpArray` to [`generate_grid`](@ref):

```julia
GridProp = @NamedTuple begin
    x  :: Vec{2, Float64}
    m  :: Float64
    mv :: Vec{2, Float64}
    v  :: Vec{2, Float64}
end

mesh = CartesianMesh(0.02, (-1, 1), (0, 3))
grid = generate_grid(SpArray, GridProp, mesh)
```

The first field, usually `x`, remains the mesh.
The other grid fields are stored as `SpArray`s and share the same sparsity pattern.

## Updating sparsity

Inactive entries are treated as structural zeros: they read as zero, and writes to inactive entries are ignored.
This also applies to broadcasts with a `SpArray` destination.
For example, `A .= B .+ C` only materializes values on the active blocks of `A`; values that would fall outside those blocks are not stored.
Because the active blocks define where values can be stored, update the sparsity pattern before using the grid in transfers:

```julia
partition = ColorPartition(grid.x)
update!(partition, particles.x)
update_sparsity!(grid, partition)
```

This is the usual CPU path when [`ColorPartition`](@ref) is already used for threaded scattering.
The partition stores the particle lists needed by `@threaded @P2G`, and the same block information can be reused to activate the sparse grid.

After that, the usual transfer macros can be used with the sparse grid:

```julia
@P2G grid=>i particles=>p weights=>ip begin
    m[i] = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
end
```

For threaded CPU scattering, pass the same partition to `@P2G` as usual.

For a `SpGrid`, call `update_sparsity!` on the whole grid rather than on each field.
This keeps all `SpArray` fields using the same active blocks.

!!! note
    `update_sparsity!` changes the active block pattern of the grid.
    Existing values in the sparse grid fields are reset to zero when the active blocks are updated.

## GPU usage

On GPU, `ColorPartition` is not used.
Move the grid, particles, and basis weights to the GPU, then update the sparse grid directly from particle positions:

```julia
grid = generate_grid(SpArray, GridProp, mesh)
weights = generate_basis_weights(Float32, BSpline(Quadratic()), grid.x, length(particles))

grid, particles, weights = (grid, particles, weights) .|> gpu

update_sparsity!(grid, particles.x)
update!(weights, particles, grid.x)

@P2G grid=>i particles=>p weights=>ip begin
    m[i] = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
end
```

For moving particles, call `update_sparsity!(grid, particles.x)` again before transfers that use the new particle positions.
This keeps the active blocks large enough for the particle support nodes used by `@P2G` and `@G2P`.

On GPU, `@P2G` uses particle-parallel kernels with atomic updates.
`SpArray` reduces the storage used by grid fields and the cost of grid-wide operations such as zeroing or broadcasts over active grid data.
The particle-to-grid scatter itself still performs work proportional to the number of particles times the number of support nodes, so `SpArray` should be viewed primarily as a memory-saving representation rather than a guaranteed P2G speedup.

## Block size

`SpArray` stores data block by block.
The block size is controlled by the `block_size_log2` option of [`CartesianMesh`](@ref), and the same block decomposition is used by [`ColorPartition`](@ref):

By default, `CartesianMesh` uses `block_size_log2=Val(Tesserae.BLOCK_SIZE_LOG2)`, so the block width is `2^Tesserae.BLOCK_SIZE_LOG2` grid indices per dimension.

```julia
mesh = CartesianMesh(0.02, (-1, 1), (-1, 1); block_size_log2=Val(3))
grid = generate_grid(SpArray, GridProp, mesh)
partition = ColorPartition(grid.x)
```

Larger blocks reduce sparsity bookkeeping, while smaller blocks follow the active domain more tightly.

## API

```@docs
SpArray
```
