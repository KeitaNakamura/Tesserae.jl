# -----------------------------------------------------------------------------
#  Export
# -----------------------------------------------------------------------------

import WriteVTK

# ---- VTK ----

to_vtk_celltype(::Line2) = WriteVTK.VTKCellTypes.VTK_LINE
to_vtk_celltype(::Line3) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_EDGE
to_vtk_celltype(::Line4) = WriteVTK.VTKCellTypes.VTK_CUBIC_LINE
to_vtk_celltype(::Tri3)  = WriteVTK.VTKCellTypes.VTK_TRIANGLE
to_vtk_celltype(::Tri6)  = WriteVTK.VTKCellTypes.VTK_QUADRATIC_TRIANGLE
to_vtk_celltype(::Quad4) = WriteVTK.VTKCellTypes.VTK_QUAD
to_vtk_celltype(::Quad8) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_QUAD
to_vtk_celltype(::Quad9) = WriteVTK.VTKCellTypes.VTK_BIQUADRATIC_QUAD
to_vtk_celltype(::Tet4)  = WriteVTK.VTKCellTypes.VTK_TETRA
to_vtk_celltype(::Tet10) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_TETRA
to_vtk_celltype(::Hex8)  = WriteVTK.VTKCellTypes.VTK_HEXAHEDRON
to_vtk_celltype(::Hex20) = WriteVTK.VTKCellTypes.VTK_QUADRATIC_HEXAHEDRON
to_vtk_celltype(::Hex27) = WriteVTK.VTKCellTypes.VTK_TRIQUADRATIC_HEXAHEDRON

to_vtk_connectivity(::Line2) = SVector(1,2)
to_vtk_connectivity(::Line3) = SVector(1,2,3)
to_vtk_connectivity(::Line4) = SVector(1,2,3,4)
to_vtk_connectivity(::Tri3)  = SVector(1,2,3)
to_vtk_connectivity(::Tri6)  = SVector(1,2,3,4,6,5)
to_vtk_connectivity(::Quad4) = SVector(1,2,3,4)
to_vtk_connectivity(::Quad8) = SVector(1,2,3,4,5,6,7,8)
to_vtk_connectivity(::Quad9) = SVector(1,2,3,4,5,6,7,8,9)
to_vtk_connectivity(::Tet4)  = SVector(1,2,3,4)
to_vtk_connectivity(::Tet10) = SVector(1,2,3,4,5,8,6,7,10,9)
to_vtk_connectivity(::Hex8)  = SVector(1,2,3,4,5,6,7,8)
to_vtk_connectivity(::Hex20) = SVector(1,2,3,4,5,6,7,8,9,12,14,10,17,19,20,18,11,13,15,16)
to_vtk_connectivity(::Hex27) = SVector(1,2,3,4,5,6,7,8,9,12,14,10,17,19,20,18,11,13,15,16,23,24,22,25,21,26,27)

struct Polyline{V <: AbstractVector{<: Vec}}
    vertices::V
end

function _vtk_grid(vtk, x::Polyline; kwargs...)
    coords = vtk_format(f32(x.vertices))
    cells = [WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_POLY_LINE, 1:length(x.vertices))]
    WriteVTK.vtk_grid(vtk, coords, cells; kwargs...)
end

function _vtk_grid(vtk, x::AbstractVector{<: Vec}; kwargs...)
    coords = vtk_format(f32(x))
    npts = length(x)
    cells = [WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    WriteVTK.vtk_grid(vtk, coords, cells; kwargs...)
end
function _vtk_grid(vtk, mesh::CartesianMesh; kwargs...)
    WriteVTK.vtk_grid(vtk, maparray(x->Vec{3,Float32}(resize(x,3)), mesh); kwargs...)
end

function _vtk_grid(vtk, mesh::FEMesh; kwargs...)
    shape = cellshape(mesh)
    vtk_cells = WriteVTK.MeshCell[]
    celltype = to_vtk_celltype(shape)
    for cell in cells(mesh)
        indices = supportnodes(mesh, cell)
        push!(vtk_cells, WriteVTK.MeshCell(celltype, indices[to_vtk_connectivity(shape)]))
    end
    WriteVTK.vtk_grid(vtk, maparray(x->Vec{3,Float32}(resize(x,3)), mesh), vtk_cells; kwargs...)
end

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

"""
    openvtk(filename, x; kwargs...) -> vtk
    openvtk(f::Function, filename, x; kwargs...)

Open a VTK dataset file for `x`, which may be a `CartesianMesh`, an `FEMesh`,
or a vector of points. Fields are written with `vtk["name"] = data` and the
file is saved with [`closevtk`](@ref). The `do`-block form saves the file and
returns the written paths. Keyword arguments are passed to
`WriteVTK.vtk_grid`.
"""
openvtk(vtk, x; kwargs...) = _vtk_grid(vtk, x; kwargs...)
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

"""
    openvtm(filename) -> vtm

Open a VTK multiblock file. Blocks are added with `openvtk(vtm, x)` and the
file is saved with [`closevtm`](@ref).
"""
openvtm(args...; kwargs...) = WriteVTK.vtk_multiblock(args...; kwargs...)

"""
    openpvd(filename) -> pvd

Open a ParaView collection for a time series. Snapshots are added with
`pvd[t] = vtk` and the collection is saved with [`closepvd`](@ref).
"""
openpvd(args...; kwargs...) = WriteVTK.paraview_collection(args...; kwargs...)

"""
    closevtk(vtk)

Save a file opened with [`openvtk`](@ref) to disk.
"""
closevtk(file::WriteVTK.DatasetFile) = WriteVTK.vtk_save(file)

"""
    closevtm(vtm)

Save a file opened with [`openvtm`](@ref) to disk.
"""
closevtm(file::WriteVTK.MultiblockFile) = WriteVTK.vtk_save(file)

"""
    closepvd(pvd)

Save a collection opened with [`openpvd`](@ref) to disk.
"""
closepvd(file::WriteVTK.CollectionFile) = WriteVTK.vtk_save(file)

f32(A::AbstractArray{Float64}) = maparray(Float32, A)
f32(A::AbstractArray{Float32}) = A
f32(A::AbstractArray{<: Tensor{<: Any, Float64}}) = maparray(concrete_tensortype(eltype(A), Float32), A)
f32(A::AbstractArray{<: Tensor{<: Any, Float32}}) = A

function concrete_tensortype(::Type{TT}, ::Type{T}) where {TT <: Tensor, T}
    space = Tensorial.Space(TT)
    Tensorial.tensortype(space){T, Tensorial.tensororder(space), Tensorial.ncomponents(space)}
end

# ---- PLY ----

function write_ply(name::AbstractString, xs::AbstractArray{<: Vec})
    file = string(name, ".ply")
    n = length(xs)
    open(file, "w") do io
        println(io, "ply")
        println(io, "format binary_little_endian 1.0")
        println(io, "element vertex $n")
        println(io, "property float x")
        println(io, "property float y")
        println(io, "property float z")
        println(io, "end_header")
    end
    open(file, "a") do io
        for X in xs
            x, y, z = resize(X, 3)
            write(io, Float32(x), Float32(y), Float32(z))
        end
    end
end
