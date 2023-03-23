# vtk_grid
function WriteVTK.vtk_grid(vtk::AbstractString, x::AbstractVector{<: Vec}; kwargs...)
    coords = vtk_format(x)
    npts = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    vtk_grid(vtk, coords, cells; kwargs...)
end
function WriteVTK.vtk_grid(vtk::AbstractString, lattice::Lattice; kwargs...)
    vtk_grid(vtk, map(collect, get_axes(lattice))...; kwargs...)
end
function WriteVTK.vtk_grid(vtk::AbstractString, grid::Grid; kwargs...)
    vtk_grid(vtk, get_lattice(grid); kwargs...)
end

# add_field_data
function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractArray{<: Tensor}, name::AbstractString, loc::WriteVTK.AbstractFieldData; kwargs...)
    vtk_point_data(vtk, vtk_format(data), name)
end
vtk_format(A::AbstractArray{<: Tensor}) = _vtk_format(A)
function vtk_format(A::AbstractArray{<: Vec{2}})
    _vtk_format(map(x->[x;0], A)) # extend to 3D to draw vector in Glyph
end
function _vtk_format(A::AbstractArray{<: Tensor})
    reinterpret(reshape, eltype(eltype(A)), A)
end

# open/close
openvtk(args...; kwargs...) = vtk_grid(args...; kwargs...)
openvtm(args...; kwargs...) = vtk_multiblock(args...; kwargs...)
openpvd(args...; kwargs...) = paraview_collection(args...; kwargs...)
closevtk(file::WriteVTK.DatasetFile) = vtk_save(file)
closevtm(file::WriteVTK.MultiblockFile) = vtk_save(file)
closepvd(file::WriteVTK.CollectionFile) = vtk_save(file)
