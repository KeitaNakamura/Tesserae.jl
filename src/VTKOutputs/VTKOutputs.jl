module VTKOutputs

using Reexport
@reexport using WriteVTK
using Jams.TensorValues
using Jams.Collections
using Jams.Grids
using Jams.States
using Jams.RigidBodies

export
    vtk_points

"""
    vtk_points(filename::AbstractString, points::PointState{<: Vec})

Create VTK file to visualize `points`.
This should be used instead of calling `vtk_grid` in `WriteVTK` package.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0);

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
julia> grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0);

julia> vtkfile = vtk_grid("vtkfile", grid)
VTK file 'vtkfile.vtr' (RectilinearGrid file, open)

julia> vtk_save(vtkfile)
1-element Array{String,1}:
 "vtkfile.vtr"
```
"""
function WriteVTK.vtk_grid(vtk::AbstractString, grid::AbstractGrid)
    vtk_grid(vtk, map(collect, gridaxes(grid))...)
end

function WriteVTK.vtk_grid(vtk::AbstractString, poly::Polygon)
    coords = vtk_format(poly)
    cells = [MeshCell(VTKCellTypes.VTK_POLYGON, collect(1:length(poly)))]
    vtk_grid(vtk, coords, cells)
end

"""
    vtk_point_data(vtk, data::AbstractVector{ <:Vec}, name)

Write the vector field data to the `vtk` file.
"""
function WriteVTK.vtk_point_data(vtk::WriteVTK.DatasetFile, data::Union{PointState{<: Tensor}, AbstractVector{<: Tensor}}, name::AbstractString)
    vtk_point_data(vtk, vtk_format(data), name)
end
function WriteVTK.vtk_point_data(vtk::WriteVTK.DatasetFile, data::AbstractCollection{2}, name::AbstractString)
    vtk_point_data(vtk, collect(data), name)
end

function vtk_format(x::Union{PointState{Vec{dim, T}}, AbstractVector{Vec{dim, T}}}) where {dim, T}
    n = length(x)
    v = reinterpret(T, Array(x))
    out = zeros(T, (dim == 2 ? 3 : dim), n)
    out[1:dim, :] .= reshape(v, dim, n)
    out
end

vtk_format(data::Union{PointState{<: Tensor}, AbstractVector{<: Tensor}}) = vtk_format([Vec(Tuple(x)) for x in data])

end
