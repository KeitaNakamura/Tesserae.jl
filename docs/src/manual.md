# [Manual overview](@id manual_overview)

Tesserae is organized around the data flow of an MPM simulation.
The core objects are deliberately small: a background mesh, grid fields, particles, basis weights, and transfer macros.
Most explicit simulations follow the same loop:

```julia
mesh = CartesianMesh(...)
grid = generate_grid(GridProp, mesh)
particles = generate_particles(ParticleProp, grid.x)
weights = generate_basis_weights(BSpline(Quadratic()), grid.x, length(particles))

for step in 1:nsteps
    update!(weights, particles, grid.x)

    @P2G grid=>i particles=>p weights=>ip begin
        # scatter particle mass, momentum, and forces to the grid
    end

    for i in eachindex(grid)
        # update grid velocities and apply boundary conditions
    end

    @G2P grid=>i particles=>p weights=>ip begin
        # gather grid updates back to particles
    end
end
```

The explicit loop is built from five parts.
[Mesh](mesh.md) covers the geometric background domain separated from grid state fields.
[Grid and particle generation](generation.md) provides the state containers used by simulations.
[Basis Functions](basis.md) provides the particle-grid connectivity used by transfer macros.
[Transfer between grid and particles](@ref manual) provides the syntax for particle-to-grid and grid-to-particle updates.
[Export](export.md) writes grid and particle fields for visualization.

Larger simulations keep the same objects while changing execution or storage.
[Multi-threading](multithreading.md) adds CPU thread parallelism.
[GPU computing](gpu.md) uses GPU-backed arrays and transfer kernels.
[SpArray](sparray.md) stores grid fields sparsely on Cartesian meshes.
Implicit formulations use a different update structure; [Utilities for implicit methods](implicit_utils.md) provides the degree-of-freedom maps, sparse matrices, matrix assembly tools, and nonlinear solvers used by those methods.
