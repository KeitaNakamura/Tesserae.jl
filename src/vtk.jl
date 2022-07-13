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
function vtk_points(vtk, x::AbstractVector{<: Vec}; kwargs...)
    coords = vtk_format(x)
    npts = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    vtk_grid(vtk, coords, cells; kwargs...)
end

function vtk_points(f::Function, vtk, x; kwargs...)
    vtk = vtk_points(vtk, x; kwargs...)
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
function WriteVTK.vtk_grid(vtk::AbstractString, grid::Grid; kwargs...)
    vtk_grid(vtk, map(collect, gridaxes(grid))...; kwargs...)
end

function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractVector{<: Tensor}, name::AbstractString, ::WriteVTK.VTKPointData)
    vtk_point_data(vtk, vtk_format(data), name)
end

function vtk_format(x::AbstractVector{Vec{dim, T}}) where {dim, T}
    reshape(reinterpret(T, x), (dim, length(x)))
end

function vtk_format(data::AbstractVector{<: SymmetricSecondOrderTensor{3}})
    vtk_format([@inbounds Vec(x[1,1], x[2,2], x[3,3], x[1,2], x[2,3], x[1,3]) for x in data])
end


function defalut_output_paraview_initialize(file)
    paraview_collection(vtk_save, file)
end
function defalut_output_paraview_append(file, grid, pointstate, t, index; output_grid = false, compress = true)
    paraview_collection(file, append = true) do pvd
        vtk_multiblock(string(file, index)) do vtm
            vtk_points(vtm, pointstate.x; compress) do vtk
                σ = pointstate.σ
                ϵ = pointstate.ϵ
                vtk["velocity"] = pointstate.v
                vtk["mean stress"] = @dot_lazy mean(σ)
                vtk["pressure"] = @dot_lazy -mean(σ)
                vtk["von mises stress"] = @dot_lazy sqrt(3/2 * dev(σ) ⊡ dev(σ))
                vtk["volumetric strain"] = @dot_lazy tr(ϵ)
                vtk["deviatoric strain"] = @dot_lazy sqrt(2/3 * dev(ϵ) ⊡ dev(ϵ))
                vtk["stress"] = σ
                vtk["strain"] = ϵ
                vtk["density"] = @dot_lazy pointstate.m / pointstate.V
            end
            if output_grid
                vtk_grid(vtm, grid; compress)
            end
            pvd[t] = vtm
        end
    end
end
