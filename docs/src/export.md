# Export

In Tesserae, exporting VTK files for visualization is supported.
These functionalities are built on the [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl) package.

## VTK file

To export the mesh or particles, use `openvtk` and `closevtk` functions.
First, we prepare following `grid` and `particles`:

```@example vtk
using Tesserae #hide

GridProp = @NamedTuple begin
    x  :: Vec{2, Float64}
    m  :: Float64
    mv :: Vec{2, Float64}
    f  :: Vec{2, Float64}
end
ParticleProp = @NamedTuple begin
    x :: Vec{2, Float64}
    m :: Float64
    v :: Vec{2, Float64}
end

grid = generate_grid(GridProp, CartesianMesh(1, (0, 3), (0, 4)))
particles = generate_particles(ParticleProp, grid.x)
nothing #hide
```

The grid properties can be exported as

```@example vtk
vtk_grid = openvtk("grid", grid.x)
vtk_grid["Momentum"] = grid.mv
vtk_grid["Force"] = grid.f
closevtk(vtk_grid)
```

This can also be written using a `do` block, which automatically closes the file:

```@example vtk
openvtk("grid", grid.x) do vtk
    vtk["Momentum"] = grid.mv
    vtk["Force"] = grid.f
end
```

For `particles`, the same instruction is available:

```@example vtk
# without do block
vtk_particles = openvtk("particles", particles.x)
vtk_particles["Velocity"] = particles.v
closevtk(vtk_particles)

# with do block
openvtk("particles", particles.x) do vtk
    vtk["Velocity"] = particles.v
end
```

## Multiblock data set

Sometimes, we want to export both the grid and particles into a single dataset.
This can be done using a `vtm` file with the `openvtm` and `closevtm` functions.

```@example vtk
openvtm("grid_and_particles") do vtm
    openvtk(vtm, grid.x) do vtk
        vtk["Momentum"] = grid.mv
        vtk["Force"] = grid.f
    end
    openvtk(vtm, particles.x) do vtk
        vtk["Velocity"] = particles.x
    end
end
```

## ParaView collection file 

A ParaView collection file (`pvd`) represents a time series of VTK files.

```@example vtk
filename = "Simulation"
closepvd(openpvd(filename)) # Just create a file by `closepvd`

for (step, t) in enumerate(range(0, 10, step=0.5))

    # Simulation...

    # Reopening and closing the file at each time step allows us
    # to visualize intermediate results # without having to wait
    # for the simulation to finish.
    openpvd(filename; append=true) do pvd
        openvtm(string(filename, step)) do vtm
            openvtk(vtm, grid.x) do vtk
                vtk["Momentum"] = grid.mv
                vtk["Force"] = grid.f
            end
            openvtk(vtm, particles.x) do vtk
                vtk["Velocity"] = particles.v
            end
            pvd[t] = vtm # save to pvd!
        end
    end
end
```
