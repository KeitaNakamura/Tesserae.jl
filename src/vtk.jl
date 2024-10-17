# vtk_grid
function WriteVTK.vtk_grid(vtk::AbstractString, x::AbstractVector{<: Vec}; kwargs...)
    coords = vtk_format(f32(x))
    npts = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    vtk_grid(vtk, coords, cells; kwargs...)
end
function WriteVTK.vtk_grid(vtk::AbstractString, mesh::CartesianMesh; kwargs...)
    vtk_grid(vtk, map(f32, get_axes(mesh))...; kwargs...)
end

# add_field_data
function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractArray{Float64}, name::AbstractString, loc::WriteVTK.AbstractFieldData; kwargs...)
    WriteVTK.add_field_data(vtk, f32(data), name, loc; kwargs...)
end
function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractArray{<: Tensor}, name::AbstractString, loc::WriteVTK.AbstractFieldData; kwargs...)
    WriteVTK.add_field_data(vtk, f32(vtk_format(data)), name, loc; kwargs...)
end
vtk_format(A::AbstractArray{<: Tensor}) = _vtk_format(A)
function _vtk_format(A::AbstractArray{<: Tensor})
    reinterpret(reshape, eltype(eltype(A)), A)
end
function _vtk_format(A::AbstractArray{<: SymmetricSecondOrderTensor{2, T}}) where {T}
    _vtk_format(maparray(x->SymmetricSecondOrderTensor{3, T}(x[1,1],x[2,1],NaN,x[2,2],NaN,NaN), A))
end
function _vtk_format(A::AbstractArray{<: SymmetricSecondOrderTensor{3, T}}) where {T}
    _vtk_format(maparray(x->Vec{6, T}(x[1,1],x[2,2],x[3,3],x[1,2],x[2,3],x[3,1]), A))
end

# open/close
openvtk(args...; kwargs...) = vtk_grid(args...; kwargs...)
openvtm(args...; kwargs...) = vtk_multiblock(args...; kwargs...)
openpvd(args...; kwargs...) = paraview_collection(args...; kwargs...)
closevtk(file::WriteVTK.DatasetFile) = vtk_save(file)
closevtm(file::WriteVTK.MultiblockFile) = vtk_save(file)
closepvd(file::WriteVTK.CollectionFile) = vtk_save(file)

# f32
f32(A::AbstractArray{Float64}) = maparray(Float32, A)
f32(A::AbstractArray{Float32}) = A
f32(A::AbstractArray{<: Tensor{<: Any, Float64}}) = maparray(Tensorial.tensortype(Tensorial.Space(eltype(A))){Float32}, A)
f32(A::AbstractArray{<: Tensor{<: Any, Float32}}) = A
