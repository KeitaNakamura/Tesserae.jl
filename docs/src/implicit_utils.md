# Utilities for implicit methods

This page collects helper types and macros used by implicit tutorials, such as degree-of-freedom maps, sparse matrices, matrix assembly, and nonlinear solvers.

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
