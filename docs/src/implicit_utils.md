# Functionalities for implicit methods

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
