# Changelog

All notable changes to Tesserae.jl will be documented in this file.

## v0.7.0

v0.7.0 focuses on larger and more inspectable MPM workflows: sparse-grid GPU
execution, backend-aware local loops, clearer threaded CPU scattering, more
predictable particle generation, and better tools for understanding transfer
macros.

### Highlights

- Sparse-grid workflows now run on GPU: `SpArray` sparsity can be updated from
  particle positions, and sparse-grid transfers dispatch to GPU kernels. (#112)
- CPU scattering now uses `ThreadPartition`, a clearer replacement for
  `ColorPartition`, with faster updates and adaptive particle reordering.
  (#122, #134, #135, #136, #137, #138)
- `PoissonDiskSampling(spacing=...)` now matches the intended particles-per-cell
  density much more closely, making particle counts easier to choose from grid
  spacing. (#113)
- `@explain` now prints readable CPU reference code for transfer macros,
  including threaded transfers and matrix assembly. (#126)
- `@foreach` provides backend-aware loops over grids, particles, and active
  sparse grid nodes without abusing transfer macros as generic loop kernels.
- FEM support is no longer experimental, with a documented workflow for mesh
  generation, quadrature, assembly, and boundary conditions. (#157)
- Experimental NURBS and IGA support has been added, including rational B-spline
  control nets, geometry operations, and conversion to `IGAMesh`.
- Gmsh `.msh` files can now be read directly as physical-group meshes through
  `readmsh`, with boundary-cell reorientation against volume faces. (#146)
- P2G and basis-weight updates are faster across dense and sparse grid paths.
  (#124, #129, #131, #132, #133)

### Breaking Changes

- `create_sparse_matrix` now requires the `ndofs` keyword to be specified
  explicitly. The previous default based on the mesh dimension was removed.
- The `PoissonDiskSampling` keyword `multithreading` has been renamed to
  `threaded`. The old keyword is no longer accepted. (#113)
- `reorder_particles!` now returns a `Bool` indicating whether particles were
  reordered, and the legacy overload accepting block-to-particle index lists
  was removed. Pass a `ThreadPartition` instead. (#137)
- `CPDI` on `SpGrid` is now rejected explicitly; use a dense grid for CPDI.
  (#130)
- `feupdate!` has been removed. Use
  `update!(weights, points, geometry; measure=nothing, normal=nothing)` instead;
  `points` carries the quadrature rule.
- The finite-element mesh type has been renamed from `UnstructuredMesh` to
  `FEMesh`; the old name is no longer available. (#156)

### Behavior Changes

- `CartesianMesh` now expands axes whose upper bounds are not exactly divisible
  by the spacing. Code that assumes the old truncated axis length may see
  different mesh sizes. Pass `warn=false` to suppress the warning. (#120)
- `PoissonDiskSampling(spacing=...)` now produces particle counts much closer to
  the intended particles-per-cell (ppc) target. Simulations that depend on the
  exact old particle count or seeded particle layout may produce different
  initial particles. (#113)
- GPU basis-weight updates, transfers, `@foreach`, stencil operations, and
  `SpArray` broadcasts no longer synchronize implicitly after launching a
  kernel. Synchronize explicitly before timing or host-side work that requires
  completion. (#152)
- `newton!` now preserves the last finite iterate at `maxiter`, evaluates `J(x)`
  after `f(x)` at the same state, and restores the last accepted state after
  failed backtracking. (#140)

### Deprecations

- `ThreadPartition` replaces `ColorPartition`. The old name remains available
  as a deprecated alias, but new code should use `ThreadPartition(mesh)`. (#122)
- Prefer `supportnodes` over `neighboringnodes` for mesh and basis-weight
  support queries. `neighboringnodes` remains available as a deprecated name.
  (#155)

### Added

- Added `generate_field_meshes` for constructing consistently numbered field
  meshes that share geometry coordinates.
- Added GPU support for `SpArray` sparsity updates and sparse-grid transfers.
  (#112)
- Added `@explain` and `ExplainedCode` for inspecting transfer macro structure.
  (#126)
- Added `@foreach` for CPU/GPU loops over grids, particles, and active `SpGrid`
  nodes.
- Added threshold-controlled adaptive particle reordering via
  `reorder_particles!(...; threshold=...)`. (#137)
- Added transfer interpolation with `$(expr)` inside transfer macro right-hand
  sides, allowing an outer expression to be evaluated once before generated
  transfer loops. (#118)
- Added `@P2G_Matrix` support for `SpGrid`. (#127)
- Added `block_size_log2` configuration to `CartesianMesh` for controlling the
  block decomposition used by `ThreadPartition` and `SpArray`. (#111)
- Added support for suppressing CartesianMesh domain-covering warnings with
  `warn=false`. (#123)

### Changed

- Generalized B-spline `values1d` generation across supported degrees. (#128)
- Replaced broad vectorized grid calculations generated by transfer macros with
  explicit grid-node loops, improving support for sparse grids and GPU paths.
  (#117)
- Replaced the ProgressMeter-based `@showprogress` implementation with
  Tesserae's built-in progress display and exported the macro. (#139)

### Performance

- Optimized thread partition updates, including dense linear update paths and
  faster particle assignment bookkeeping. (#134, #135, #138)
- Optimized StructVector particle reorder buffers. (#136)
- Accelerated full-support B-spline basis-weight updates with direct
  assignments. (#129)
- Reduced P2G scatter overhead for `SpGrid` through improved write indexing and
  faster sparse storage writes. (#131, #132)
- Hoisted particle-only products from P2G RHS expressions. (#133)
- Shared basis-weight loads inside `@G2P2G`. (#124)

### Fixed

- Fixed `SpArray` broadcast materialization and display behavior. (#116)

### Dependencies

- Updated compatibility for PoissonDiskSampling.jl v1 and Tensorial.jl 0.20.3.
  (#113, #121)
- Removed the ProgressMeter.jl dependency. (#139)
- Added JuliaFormatter.jl for formatted `@explain` output. (#126)
