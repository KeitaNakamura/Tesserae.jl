# vtk_grid
function WriteVTK.vtk_grid(vtk::AbstractString, x::AbstractVector{<: Vec}; kwargs...)
    coords = vtk_format(f32(x))
    npts = length(x)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:npts]
    vtk_grid(vtk, coords, cells; kwargs...)
end
function WriteVTK.vtk_grid(vtk::AbstractString, lattice::Lattice; kwargs...)
    vtk_grid(vtk, map(f32, get_axes(lattice))...; kwargs...)
end

# add_field_data
function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractArray{Float64}, name::AbstractString, loc::WriteVTK.AbstractFieldData; kwargs...)
    WriteVTK.add_field_data(vtk, f32(data), name, loc; kwargs...)
end
function WriteVTK.add_field_data(vtk::WriteVTK.DatasetFile, data::AbstractArray{<: Tensor}, name::AbstractString, loc::WriteVTK.AbstractFieldData; kwargs...)
    WriteVTK.add_field_data(vtk, vtk_format(f32(data)), name, loc; kwargs...)
end
vtk_format(A::AbstractArray{<: Tensor}) = _vtk_format(A)
function _vtk_format(A::AbstractArray{<: Tensor})
    reinterpret(reshape, eltype(eltype(A)), A)
end
function vtk_format(A::AbstractArray{<: Vec{2}})
    _vtk_format(Vec2ToVec3(A)) # extend to 3D to draw vector in Glyph
end
struct Vec2ToVec3{T, N, A <: AbstractArray{Vec{2, T}, N}} <: AbstractArray{Vec{3, T}, N}
    array::A
end
Base.IndexStyle(::Type{<: Vec2ToVec3}) = IndexLinear()
Base.size(x::Vec2ToVec3) = size(x.array)
Base.getindex(x::Vec2ToVec3, i::Integer) = (@_propagate_inbounds_meta; [x.array[i];0])

# open/close
openvtk(args...; kwargs...) = vtk_grid(args...; kwargs...)
openvtm(args...; kwargs...) = vtk_multiblock(args...; kwargs...)
openpvd(args...; kwargs...) = paraview_collection(args...; kwargs...)
closevtk(file::WriteVTK.DatasetFile) = vtk_save(file)
closevtm(file::WriteVTK.MultiblockFile) = vtk_save(file)
closepvd(file::WriteVTK.CollectionFile) = vtk_save(file)

# f32
struct SingleMapArray{T, N, F, U, A <: AbstractArray{U, N}} <: AbstractArray{T, N}
    f::F
    parent::A
end
function SingleMapArray(f::F, parent::A) where {F, N, U, A <: AbstractArray{U, N}}
    T = Base._return_type(f, Tuple{U})
    SingleMapArray{T, N, F, U, A}(f, parent)
end
Base.size(A::SingleMapArray) = size(A.parent)
Base.IndexStyle(::Type{<: SingleMapArray{<: Any, <: Any, <: Any, <: Any, A}}) where {A} = IndexStyle(A)
@inline function Base.getindex(A::SingleMapArray, i...)
    @boundscheck checkbounds(A, i...)
    @inbounds A.f(A.parent[i...])
end

f32(A::AbstractArray{Float64}) = SingleMapArray(x->convert(Float32, x), A)
f32(A::AbstractArray{Float32}) = A
f32(A::AbstractArray{<: Tensor{<: Any, Float64}}) = SingleMapArray(x->Tensorial.tensortype(Space(eltype(A))){Float32}(Tuple(x)), A)
f32(A::AbstractArray{<: Tensor{<: Any, Float32}}) = A
