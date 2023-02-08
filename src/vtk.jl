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

function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractVector{<: Tensor}, name::AbstractString, ::WriteVTK.VTKPointData)
    vtk_point_data(vtk, vtk_format(data), name)
end

function vtk_format(x::AbstractVector{Vec{dim, T}}) where {dim, T}
    reshape(reinterpret(T, x), (dim, length(x)))
end
function vtk_format(data::AbstractVector{<: SymmetricSecondOrderTensor{3}})
    vtk_format([@inbounds Vec(x[1,1], x[2,2], x[3,3], x[1,2], x[2,3], x[1,3]) for x in data])
end

openvtk(args...; kwargs...) = vtk_grid(args...; kwargs...)
openvtm(args...; kwargs...) = vtk_multiblock(args...; kwargs...)
openpvd(args...; kwargs...) = paraview_collection(args...; kwargs...)
closevtk(file::WriteVTK.DatasetFile) = vtk_save(file)
closevtm(file::WriteVTK.MultiblockFile) = vtk_save(file)
closepvd(file::WriteVTK.CollectionFile) = vtk_save(file)
