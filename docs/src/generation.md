# Grid and particle generation

Grid and particle generation starts from the fields that each object should carry.
In Tesserae, those fields are described by a property type, written either as a `struct` or an `@NamedTuple`.

For grids, the property type defines fields stored at background nodes.
For particles, it defines fields stored on material points.
The first field is reserved for position: grid positions are backed by the mesh coordinates, while particle positions are filled by the sampling algorithm used in `generate_particles`.

```@docs
generate_grid
generate_particles
GridSampling
PoissonDiskSampling
```

![](https://github.com/user-attachments/assets/1ed7db73-cce2-47b2-84fc-9e895b9a89a7)
