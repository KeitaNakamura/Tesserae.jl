import ..Tesserae
using ..Tesserae: CartesianMesh, generate_grid, spacing, spacing_inv

#################
# StaggeredGrid #
#################

function Tesserae.generate_grid(::Type{Arr}, loc::Cell, ::Type{CellProp}, mesh::CartesianMesh{dim}) where {Arr, CellProp, dim}
    h = spacing(mesh)
    h⁻¹ = spacing_inv(mesh)
    axes = mesh.axes
    cellaxes = map(ax -> ax[1:end-1] .+ h/2, axes)
    StructArrays.replace_storage(generate_grid(Arr, CellProp, CartesianMesh(cellaxes, h, h⁻¹))) do a
        a isa CartesianMesh && return a
        StencilArray(loc, a)
    end
end
Tesserae.generate_grid(loc::Cell, ::Type{CellProp}, mesh::CartesianMesh) where {CellProp} = generate_grid(Array, loc, CellProp, mesh)

function Tesserae.generate_grid(::Type{Arr}, loc::Face, ::Type{FaceProp}, mesh::CartesianMesh{dim}) where {Arr, FaceProp, dim}
    h = spacing(mesh)
    h⁻¹ = spacing_inv(mesh)
    axes = mesh.axes
    cellaxes = map(ax -> ax[1:end-1] .+ h/2, axes)
    StructArrays.replace_storage(generate_grid(Arr, FaceProp, CartesianMesh(ntuple(j -> j==loc.axis ? axes[j] : cellaxes[j], Val(dim)), h, h⁻¹))) do a
        a isa CartesianMesh && return a
        StencilArray(loc, a)
    end
end
Tesserae.generate_grid(loc::Face, ::Type{FaceProp}, mesh::CartesianMesh) where {FaceProp} = generate_grid(Array, loc, FaceProp, mesh)

#########
# utils #
#########

function inner(mesh::CartesianMesh; pad::Int)
    h = spacing(mesh)
    h⁻¹ = spacing_inv(mesh)
    axes = map(mesh.axes) do ax
        @view ax[begin+pad:end-pad]
    end
    CartesianMesh(axes, h, h⁻¹)
end

Tesserae.fillzero!(x::StencilArray) = (Tesserae.fillzero!(parent(x)); x)

Tesserae.hybrid(A::StencilArray) = Tesserae.HybridArray(parent(A), Tesserae.flatten(parent(A)), Tesserae.get_device(A))

function Adapt.adapt_structure(to, A::StencilArray)
    StencilArray(getlocation(A), adapt(to, parent(A)))
end
