"""
    vtk_points(filename::AbstractString, points::AbstractVector{<: Vec})

Create VTK file to visualize `points`.
This should be used instead of calling `vtk_grid` in `WriteVTK` package.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0);

julia> pointstate = generate_pointstate((x, y) -> (x-5)^2 + (y-5)^2 < 3^2, grid, n = 4);

julia> vtkfile = vtk_points("vtkfile", pointstate.x)
VTK file 'vtkfile.vtu' (UnstructuredGrid file, open)

julia> vtk_save(vtkfile)
1-element Vector{String}:
 "vtkfile.vtu"
```
"""
function vtk_points(vtk, x::AbstractVector{<: Vec})
    coords = vtk_format(x)
    npts = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    vtk_grid(vtk, coords, cells)
end

function vtk_points(f::Function, vtk, x)
    vtk = vtk_points(vtk, x)
    local outfile::Vector{String}
    try
        f(vtk)
    finally
        outfile = vtk_save(vtk)
    end
    outfile
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
1-element Vector{String}:
 "vtkfile.vtr"
```
"""
function WriteVTK.vtk_grid(vtk::AbstractString, grid::Grid)
    vtk_grid(vtk, map(collect, gridaxes(grid))...)
end

function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractVector{<: Tensor}, name::AbstractString, ::WriteVTK.VTKPointData)
    vtk_point_data(vtk, vtk_format(data), name)
end

function vtk_format(x::AbstractVector{Vec{dim, T}}) where {dim, T}
    n = length(x)
    v = reinterpret(T, Array(x))
    out = zeros(T, (dim == 2 ? 3 : dim), n)
    out[1:dim, :] .= reshape(v, dim, n)
    out
end

function vtk_format(data::AbstractVector{<: SymmetricSecondOrderTensor{3}})
    vtk_format([@inbounds Vec(x[1,1], x[2,2], x[3,3], x[1,2], x[2,3], x[1,3]) for x in data])
end
