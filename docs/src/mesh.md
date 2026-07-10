# Mesh

A background mesh and a background grid are often used interchangeably in MPM literature, but Tesserae keeps them separate.
A mesh stores the geometry of the background domain, while a grid is a state container generated from that mesh.
Grid fields such as mass, momentum, velocity, and force live on the grid; coordinates and spacing come from the mesh.

A simulation typically shares one mesh between the grid, particles, and basis weights.
This keeps coordinates and particle-grid connectivity consistent across the calculation.

!!! info
    MPM workflows currently use [`CartesianMesh`](@ref). [`FEMesh`](@ref) is used for finite-element calculations; [`IGAMesh`](@ref) is used for isogeometric calculations. See [Finite element calculations](fem.md) and [Isogeometric analysis calculations](iga.md).

## Cartesian mesh

```@docs
CartesianMesh
spacing
volume
extract(::CartesianMesh{dim}, ::Vararg{Tuple{Real, Real}, dim}) where {dim}
isinside
findcell
```
