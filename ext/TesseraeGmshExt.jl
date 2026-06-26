module TesseraeGmshExt

using Tesserae
using Gmsh
using StaticArrays

const GMSH_COORD_DIM = 3

function from_gmsh_shape(name::String)
    name == "Line 2" && return Tesserae.Line2()
    name == "Line 3" && return Tesserae.Line3()
    name == "Line 4" && return Tesserae.Line4()
    name == "Triangle 3" && return Tesserae.Tri3()
    name == "Triangle 6" && return Tesserae.Tri6()
    name == "Tetrahedron 4" && return Tesserae.Tet4()
    name == "Tetrahedron 10" && return Tesserae.Tet10()
    name == "Quadrilateral 4" && return Tesserae.Quad4()
    name == "Quadrilateral 8" && return Tesserae.Quad8()
    name == "Quadrilateral 9" && return Tesserae.Quad9()
    name == "Hexahedron 8" && return Tesserae.Hex8()
    name == "Hexahedron 20" && return Tesserae.Hex20()
    name == "Hexahedron 27" && return Tesserae.Hex27()
    error("\"$name\" is not supported yet")
end

from_gmsh_connectivity(::Tesserae.Line2) = SVector(1, 2)
from_gmsh_connectivity(::Tesserae.Line3) = SVector(1, 2, 3)
from_gmsh_connectivity(::Tesserae.Line4) = SVector(1, 2, 3, 4)
from_gmsh_connectivity(::Tesserae.Tri3) = SVector(1, 2, 3)
from_gmsh_connectivity(::Tesserae.Tri6) = SVector(1, 2, 3, 4, 6, 5)
from_gmsh_connectivity(::Tesserae.Tet4) = SVector(1, 2, 3, 4)
from_gmsh_connectivity(::Tesserae.Tet10) = SVector(1, 2, 3, 4, 5, 7, 8, 6, 9, 10)
from_gmsh_connectivity(::Tesserae.Quad4) = SVector(1, 2, 3, 4)
from_gmsh_connectivity(::Tesserae.Quad8) = SVector(1, 2, 3, 4, 5, 6, 7, 8)
from_gmsh_connectivity(::Tesserae.Quad9) = SVector(1, 2, 3, 4, 5, 6, 7, 8, 9)
from_gmsh_connectivity(::Tesserae.Hex8) = SVector(1, 2, 3, 4, 5, 6, 7, 8)
from_gmsh_connectivity(::Tesserae.Hex20) = SVector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
from_gmsh_connectivity(::Tesserae.Hex27) = SVector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)

"""
    read_gmsh_nodes()

Read global nodes from the current Gmsh model and return `(nodes, nodeindices)`.
`nodes` is a `Vector{Vec{dim,Float64}}`, and `nodeindices` maps each Gmsh
node tag to the corresponding Tesserae node index.
"""
function read_gmsh_nodes()
    node_tags, coords, _parametric_coords = Gmsh.gmsh.model.mesh.getNodes()
    dim = Int(Gmsh.gmsh.model.getDimension())
    nodes = Vector{Vec{dim, Float64}}(undef, length(node_tags))
    nodeindices = Dict{eltype(node_tags), Int}()
    sizehint!(nodeindices, length(node_tags))
    @inbounds for i in eachindex(node_tags)
        offset = GMSH_COORD_DIM * (i - 1)
        nodes[i] = Vec{dim, Float64}(ntuple(j -> coords[offset + j], dim))
        nodeindices[node_tags[i]] = i
    end
    nodes, nodeindices
end

"""
    read_gmsh_elements(dim, entity_tag, nodeindices)

Read elements in a Gmsh entity and return a dictionary from Tesserae cell shape
to connectivities. The returned connectivities use Tesserae node indices and
Tesserae local node ordering.
"""
function read_gmsh_elements(dim, entity_tag, nodeindices)
    element_types, _element_tags, node_tags = Gmsh.gmsh.model.mesh.getElements(dim, entity_tag)
    elements = Dict{Tesserae.Shape, Vector}()
    for (element_type, tags) in zip(element_types, node_tags)
        element_name, element_dim, _order, numnodes, _local_coords, _num_primary_nodes = Gmsh.gmsh.model.mesh.getElementProperties(element_type)
        @assert dim == element_dim

        shape = from_gmsh_shape(element_name)
        @assert numnodes == Tesserae.nlocalnodes(shape)

        permutation = from_gmsh_connectivity(shape)
        connectivities = get!(elements, shape) do
            SVector{Tesserae.nlocalnodes(shape), Int}[]
        end

        @inbounds for offset in 0:numnodes:length(tags)-numnodes
            connectivity = SVector(ntuple(i -> nodeindices[tags[offset + permutation[i]]], numnodes))
            push!(connectivities, connectivity)
        end
    end
    elements
end

"""
    read_gmsh_physical_group(dim, physical_tag, nodes, nodeindices)

Read a Gmsh physical group and return `name => mesh`, where `mesh` is an
`UnstructuredMesh` built from the group's elements.
"""
function read_gmsh_physical_group(dim, physical_tag, nodes, nodeindices)
    elements = Dict{Tesserae.Shape, Vector}()
    entity_tags = Gmsh.gmsh.model.getEntitiesForPhysicalGroup(dim, physical_tag)
    for entity_tag in entity_tags
        entity_elements = read_gmsh_elements(dim, entity_tag, nodeindices)
        for (shape, connectivities) in entity_elements
            append!(get!(elements, shape, eltype(connectivities)[]), connectivities)
        end
    end

    name = Gmsh.gmsh.model.getPhysicalName(dim, physical_tag)
    if length(elements) != 1
        shapes = join(string.(collect(keys(elements))), ", ")
        error("physical group \"$name\" must contain exactly one cell shape; found $(length(elements)): $shapes")
    end
    shape, connectivities = only(elements)
    name => Tesserae.UnstructuredMesh(shape, nodes, connectivities)
end

function Tesserae.readmsh(filename::AbstractString; gmsh_argv=String[])
    initialized = Gmsh.initialize(gmsh_argv; finalize_atexit=false)
    try
        Gmsh.gmsh.open(filename)
        nothing
    finally
        initialized && Gmsh.finalize()
    end
end

end
