"""
    readmsh(filename::AbstractString; gmsh_argv=String[], reorient_boundary=true)

Read a Gmsh `.msh` file.

This method is provided by the Gmsh extension. Load `Gmsh` before calling it:

```julia
using Tesserae
using Gmsh

meshes = readmsh("mesh.msh")
```

Returns a `Dict{String,UnstructuredMesh}` keyed by physical group name.
Each physical group is read as one `UnstructuredMesh` and must contain exactly
one supported element shape. Unnamed physical groups use keys of the form
`"physical_group[dim,tag]"`.

When `reorient_boundary` is `true`, boundary cells that match exactly one
volume face are reordered to follow the outward volume face orientation.
Boundary cells that do not match a volume face, or match multiple volume faces,
are left unchanged and reported with a warning.
"""
function readmsh end
