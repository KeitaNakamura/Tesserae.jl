# Mesh

Meshes define the background coordinates used by grids, particles, and basis weights.
In a typical MPM simulation, the mesh is created first, then `generate_grid`, `generate_particles`, and `generate_basis_weights` are built from it.

!!! info
    Currently, only the Cartesian mesh is available.

## Cartesian mesh

```@docs
CartesianMesh
spacing
volume
extract(::CartesianMesh{dim}, ::Vararg{Tuple{Real, Real}, dim}) where {dim}
isinside
findcell
```
