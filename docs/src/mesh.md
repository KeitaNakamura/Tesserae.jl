# Mesh

!!! info
    Currently, only the Cartesian mesh is available.

## Cartesian mesh

```@docs
CartesianMesh
spacing
volume
extract(::CartesianMesh{dim}, ::Vararg{Tuple{Real, Real}, dim}) where {dim}
isinside
whichcell
```
