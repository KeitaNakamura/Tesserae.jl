module VTKOutputs

using Reexport
@reexport using WriteVTK
using Jams.TensorValues
using Jams.Grids
using Jams.States

export
    vtk_points

"""
    vtk_points(filename::AbstractString, points::PointState{<: Vec})

Create VTK file to visualize `points`.
This should be used instead of calling `vtk_grid` in `WriteVTK` package.

# Examples
```jldoctest
julia> grid = CartesianGrid(1.0, (11, 11));

julia> xₚ, = generate_pointstates((x, y) -> (x-5)^2 + (y-5)^2 < 3^2, grid, n = 4);

julia> vtkfile = vtk_points("vtkfile", xₚ)
VTK file 'vtkfile.vtu' (UnstructuredGrid file, open)

julia> vtk_save(vtkfile)
1-element Array{String,1}:
 "vtkfile.vtu"
```
"""
function vtk_points(vtk, x::PointState{<: Vec})
    coords = vtk_format(x)
    npts = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    vtk_grid(vtk, coords, cells)
end

function vtk_points(f::Function, vtk, x)
    vtk = vtk_points(vtk, x)
    f(vtk)
end

"""
    vtk_grid(filename::AbstractString, grid::Grid)

Create a structured VTK grid from a `Grid`.

# Examples
```jldoctest
julia> grid = CartesianGrid(1.0, (11, 11));

julia> vtkfile = vtk_grid("vtkfile", grid)
VTK file 'vtkfile.vtr' (RectilinearGrid file, open)

julia> vtk_save(vtkfile)
1-element Array{String,1}:
 "vtkfile.vtr"
```
"""
function WriteVTK.vtk_grid(vtk::AbstractString, grid::AbstractGrid)
    vtk_grid(vtk, gridaxes(grid)...)
end

"""
    vtk_point_data(vtk, data::AbstractVector{ <:Vec}, name)

Write the vector field data to the `vtk` file.
"""
function WriteVTK.vtk_point_data(vtk::WriteVTK.DatasetFile, data::PointState{<: Vec}, name::AbstractString)
    vtk_point_data(vtk, vtk_format(data), name)
end

function vtk_format(x::PointState{<: Vec{dim, T}}) where {dim, T}
    n = length(x)
    v = reinterpret(T, Array(x))
    out = zeros(T, (dim == 2 ? 3 : dim), n)
    out[1:dim, :] .= reshape(v, dim, n)
    out
end

end
