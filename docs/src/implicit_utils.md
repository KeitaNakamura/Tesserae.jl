# Utilities for implicit methods

Implicit MPM formulations often need degree-of-freedom maps, sparse matrices, matrix assembly, and nonlinear iteration.
Tesserae provides small utilities for those parts of an implicit formulation.

## Degrees of freedom (DoF) mapping

```@docs
DofMap
```

## Sparse matrix

```@docs
create_sparse_matrix
extract(::AbstractMatrix, ::DofMap)
```

## Assembly of global matrix

```@docs
@P2G_Matrix
```

## Solvers

```@docs
Tesserae.newton!
```
