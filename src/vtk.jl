import WriteVTK

# vtk_grid
function vtk_particles(vtk, x::AbstractVector{<: Vec}; kwargs...)
    coords = vtk_format(f32(x))
    npts = length(x)
    cells = [WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    WriteVTK.vtk_grid(vtk, coords, cells; kwargs...)
end
function vtk_mesh(vtk, mesh::CartesianMesh; kwargs...)
    WriteVTK.vtk_grid(vtk, maparray(x->Vec{3,Float32}(resize(x,3)), mesh); kwargs...)
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
openvtk(vtk, mesh::AbstractMesh; kwargs...) = vtk_mesh(vtk, mesh; kwargs...)
openvtk(vtk, particles::AbstractVector{<:Vec}; kwargs...) = vtk_particles(vtk, particles; kwargs...)
function openvtk(f::Function, args...; kwargs...)
    vtk = openvtk(args...; kwargs...)
    local outfiles
    try
        f(vtk)
    finally
        outfiles = close(vtk)
    end
    outfiles::Vector{String}
end
openvtm(args...; kwargs...) = WriteVTK.vtk_multiblock(args...; kwargs...)
openpvd(args...; kwargs...) = WriteVTK.paraview_collection(args...; kwargs...)
closevtk(file::WriteVTK.DatasetFile) = WriteVTK.vtk_save(file)
closevtm(file::WriteVTK.MultiblockFile) = WriteVTK.vtk_save(file)
closepvd(file::WriteVTK.CollectionFile) = WriteVTK.vtk_save(file)

# f32
f32(A::AbstractArray{Float64}) = maparray(Float32, A)
f32(A::AbstractArray{Float32}) = A
f32(A::AbstractArray{<: Tensor{<: Any, Float64}}) = maparray(Tensorial.tensortype(Tensorial.Space(eltype(A))){Float32}, A)
f32(A::AbstractArray{<: Tensor{<: Any, Float32}}) = A
