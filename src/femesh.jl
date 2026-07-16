"""
    FEMesh(shape, nodes, cellsupports)
    FEMesh(cartesian_mesh)
    FEMesh(shape, cartesian_mesh)

Create a finite-element mesh.

`shape` is the cell shape, `nodes` stores the nodal coordinates, and
`cellsupports` stores the node indices of each cell. `cells(mesh)` iterates over
cell indices, while `supportnodes(mesh, cell)` returns the nodes used by one
cell.

The Cartesian constructors are convenience constructors for structured test
meshes and examples. `FEMesh(cartesian_mesh)` uses the default first-order cell
shape for the dimension: `Line2`, `Quad4`, or `Hex8`. Passing an explicit
`shape` allows triangular, tetrahedral, and higher-order cells to be generated
from the Cartesian grid.

`FEMesh` is used by the finite-element workflow. Use
`generate_particles` to create quadrature points,
[`generate_basis_weights`](@ref) to create element-local basis storage, and
`update!` to fill the element-local basis data.
"""
struct FEMesh{S <: Shape, dim, T, L, V <: AbstractVector{Vec{dim, T}}} <: AbstractMesh{dim, T, 1}
    shape::S
    nodes::V
    cellsupports::Vector{SVector{L, Int}}
    usednodes::Vector{Int}
end

function FEMesh(shape::S, nodes::V, cellsupports::Vector{SVector{L, Int}}) where {S <: Shape, dim, T, L, V <: AbstractVector{Vec{dim, T}}}
    FEMesh{S, dim, T, L, V}(shape, nodes, cellsupports, _collect_supportnodes(cellsupports))
end

Base.size(mesh::FEMesh) = size(mesh.nodes)
Base.IndexStyle(::Type{<: FEMesh}) = IndexLinear()

@inline function Base.getindex(mesh::FEMesh, i::Int)
    @_propagate_inbounds_meta
    mesh.nodes[i]
end
@inline function Base.setindex!(mesh::FEMesh, x, i::Int)
    @_propagate_inbounds_meta
    mesh.nodes[i] = x
end

cellshape(mesh::FEMesh) = mesh.shape
ncells(mesh::FEMesh) = length(mesh.cellsupports)
cells(mesh::FEMesh) = eachindex(mesh.cellsupports)

"""
    supportnodes(mesh::FEMesh)
    supportnodes(mesh::FEMesh, cell::Int)

Return the sorted node indices used by `mesh`, or the local support node
indices of `cell`.
"""
@inline supportnodes(mesh::FEMesh) = mesh.usednodes
@inline supportnodes(mesh::FEMesh, cell::Int) = (@_propagate_inbounds_meta; mesh.cellsupports[cell])

FEMesh(mesh::CartesianMesh) = FEMesh(default_cellshape(mesh), mesh)
default_cellshape(::CartesianMesh{1}) = Line2()
default_cellshape(::CartesianMesh{2}) = Quad4()
default_cellshape(::CartesianMesh{3}) = Hex8()

function FEMesh(shape::Shape{dim}, mesh::CartesianMesh{dim}) where {dim}
    mesh′ = adapt_mesh(get_order(shape), mesh)

    cellranges = _cellnodes_ranges(get_order(shape), size(mesh).-1)
    dims = (length(_cellnodes_connectivities(shape, first(cellranges))), length(cellranges))
    connecitivies = Matrix{SVector{nlocalnodes(shape), Int}}(undef, dims)
    @inbounds for (j, range) in enumerate(cellranges)
        conns = _cellnodes_connectivities(shape, range)
        for (i, conn) in enumerate(conns)
            inds = LinearIndices(mesh′)[conn]
            connecitivies[i,j] = inds
        end
    end

    nodeindices = zeros(Int, size(mesh′)) # handle Serendipity cell
    @inbounds for conn in connecitivies
        nodeindices[conn] .= 1
    end
    count = 0
    @inbounds for i in eachindex(nodeindices)
        if nodeindices[i] != 0
            nodeindices[i] = (count+=1)
        end
    end

    @inbounds for (i, conn) in enumerate(connecitivies)
        connecitivies[i] = nodeindices[conn]
    end

    FEMesh(shape, mesh′[findall(>(0), nodeindices)], vec(connecitivies))
end
_cellnodes_ranges(::Order{1}, cellsize::Dims) = maparray(I -> I:(I+oneunit(I)), CartesianIndices(cellsize))
_cellnodes_ranges(::Order{2}, cellsize::Dims) = maparray(I -> (2I-oneunit(I)):(2I+oneunit(I)), CartesianIndices(cellsize))
@inline _cellnodes_connectivities(::Line2, CI::CartesianIndices{1}) = @inbounds (SVector(CI[1], CI[2]),)
@inline _cellnodes_connectivities(::Line3, CI::CartesianIndices{1}) = @inbounds (SVector(CI[1], CI[3], CI[2]),)
@inline _cellnodes_connectivities(::Quad4, CI::CartesianIndices{2}) = @inbounds (SVector(CI[1,1], CI[2,1], CI[2,2], CI[1,2]),)
@inline _cellnodes_connectivities(::Quad8, CI::CartesianIndices{2}) = @inbounds (SVector(CI[1,1], CI[3,1], CI[3,3], CI[1,3], CI[2,1], CI[3,2], CI[2,3], CI[1,2]),)
@inline _cellnodes_connectivities(::Quad9, CI::CartesianIndices{2}) = @inbounds (SVector(CI[1,1], CI[3,1], CI[3,3], CI[1,3], CI[2,1], CI[3,2], CI[2,3], CI[1,2], CI[2,2]),)
@inline _cellnodes_connectivities(::Hex8,  CI::CartesianIndices{3}) = @inbounds (SVector(CI[1,1,1], CI[2,1,1], CI[2,2,1], CI[1,2,1], CI[1,1,2], CI[2,1,2], CI[2,2,2], CI[1,2,2]),)
@inline _cellnodes_connectivities(::Hex20, CI::CartesianIndices{3}) = @inbounds (SVector(CI[1,1,1], CI[3,1,1], CI[3,3,1], CI[1,3,1], CI[1,1,3], CI[3,1,3], CI[3,3,3], CI[1,3,3], CI[2,1,1], CI[1,2,1], CI[1,1,2], CI[3,2,1], CI[3,1,2], CI[2,3,1], CI[3,3,2], CI[1,3,2], CI[2,1,3], CI[1,2,3], CI[3,2,3], CI[2,3,3]),)
@inline _cellnodes_connectivities(::Hex27, CI::CartesianIndices{3}) = @inbounds (SVector(CI[1,1,1], CI[3,1,1], CI[3,3,1], CI[1,3,1], CI[1,1,3], CI[3,1,3], CI[3,3,3], CI[1,3,3], CI[2,1,1], CI[1,2,1], CI[1,1,2], CI[3,2,1], CI[3,1,2], CI[2,3,1], CI[3,3,2], CI[1,3,2], CI[2,1,3], CI[1,2,3], CI[3,2,3], CI[2,3,3], CI[2,2,1], CI[2,1,2], CI[1,2,2], CI[3,2,2], CI[2,3,2], CI[2,2,3], CI[2,2,2]),)
@inline _cellnodes_connectivities(::Tri3,  CI::CartesianIndices{2}) = @inbounds (SVector(CI[1,1], CI[2,1], CI[1,2]), SVector(CI[2,2], CI[1,2], CI[2,1]))
@inline _cellnodes_connectivities(::Tri6,  CI::CartesianIndices{2}) = @inbounds (SVector(CI[1,1], CI[3,1], CI[1,3], CI[2,1], CI[1,2], CI[2,2]), SVector(CI[3,3], CI[1,3], CI[3,1], CI[2,3], CI[3,2], CI[2,2]))
@inline _cellnodes_connectivities(::Tet4,  CI::CartesianIndices{3}) = @inbounds (SVector(CI[1,1,1], CI[2,1,1], CI[2,2,1], CI[2,2,2]), SVector(CI[1,1,1], CI[2,1,2], CI[2,1,1], CI[2,2,2]), SVector(CI[1,1,1], CI[2,2,1], CI[1,2,1], CI[2,2,2]), SVector(CI[1,1,1], CI[1,2,1], CI[1,2,2], CI[2,2,2]), SVector(CI[1,1,1], CI[1,2,2], CI[1,1,2], CI[2,2,2]), SVector(CI[1,1,1], CI[1,1,2], CI[2,1,2], CI[2,2,2]))
@inline _cellnodes_connectivities(::Tet10, CI::CartesianIndices{3}) = @inbounds (SVector(CI[1,1,1], CI[3,1,1], CI[3,3,1], CI[3,3,3], CI[2,1,1], CI[2,2,1], CI[2,2,2], CI[3,2,1], CI[3,3,2], CI[3,2,2]), SVector(CI[1,1,1], CI[3,1,3], CI[3,1,1], CI[3,3,3], CI[2,1,2], CI[2,1,1], CI[2,2,2], CI[3,1,2], CI[3,2,2], CI[3,2,3]), SVector(CI[1,1,1], CI[3,3,1], CI[1,3,1], CI[3,3,3], CI[2,2,1], CI[1,2,1], CI[2,2,2], CI[2,3,1], CI[2,3,2], CI[3,3,2]), SVector(CI[1,1,1], CI[1,3,1], CI[1,3,3], CI[3,3,3], CI[1,2,1], CI[1,2,2], CI[2,2,2], CI[1,3,2], CI[2,3,3], CI[2,3,2]), SVector(CI[1,1,1], CI[1,3,3], CI[1,1,3], CI[3,3,3], CI[1,2,2], CI[1,1,2], CI[2,2,2], CI[1,2,3], CI[2,2,3], CI[2,3,3]), SVector(CI[1,1,1], CI[1,1,3], CI[3,1,3], CI[3,3,3], CI[1,1,2], CI[2,1,2], CI[2,2,2], CI[2,1,3], CI[3,2,3], CI[2,2,3]))

adapt_mesh(::Order{1}, mesh::CartesianMesh) = mesh
function adapt_mesh(::Order{2}, mesh::CartesianMesh{dim}) where {dim}
    CartesianMesh(ntuple(Val(dim)) do d
        xmin = get_xmin(mesh)[d]
        xmax = get_xmax(mesh)[d]
        h = spacing(mesh)
        xmin:(h/2):xmax
    end...; block_size_log2=Val(block_size_log2(mesh)))
end

function extract(mesh::FEMesh, nodeindices::AbstractVector{Int})
    shape = cellshape(mesh)
    supports = SVector{nlocalnodes(shape), Int}[]
    for cell in cells(mesh)
        indices = supportnodes(mesh, cell)
        if all(in.(indices, Ref(nodeindices)))
            push!(supports, indices)
        end
    end
    FEMesh(shape, mesh.nodes, supports)
end

function extract_face(mesh::FEMesh, nodeindices::AbstractVector{Int})
    shape = faceshape(cellshape(mesh))
    supports = SVector{nlocalnodes(shape), Int}[]
    for cell in cells(mesh)
        indices = supportnodes(mesh, cell)
        for conn in faces(cellshape(mesh))
            faceindices = indices[conn]
            if all(in.(faceindices, Ref(nodeindices)))
                push!(supports, faceindices)
            end
        end
    end
    FEMesh(shape, mesh.nodes, unique(supports))
end

function Base.merge!(dest::FEMesh{S}, src::FEMesh{S}) where {S}
    dest.nodes isa Vector || throw(ArgumentError("merge! requires a resizable Vector of nodes"))
    isapproxzero(x) = dot(x,x) < eps(eltype(x))

    nodemap = Vector{Int}(undef, length(src))
    @threaded for i in eachindex(src, nodemap)
        x = src[i]
        index = findfirst(y -> isapproxzero(x-y), dest)
        nodemap[i] = ifelse(isnothing(index), 0, index)
    end
    for i in eachindex(src, nodemap)
        if nodemap[i] == 0
            push!(dest.nodes, src[i])
            nodemap[i] = length(dest)
        end
    end

    sortedinds_src = map(inds -> sort(nodemap[inds]), src.cellsupports)
    sortedinds_dst = map(sort, dest.cellsupports)
    for cell in cells(src)
        inds_src = sortedinds_src[cell]
        if all(inds_dest -> inds_src !== inds_dest, sortedinds_dst)
            push!(dest.cellsupports, nodemap[supportnodes(src, cell)])
        end
    end
    _refresh_supportnodes!(dest)
end
function Base.merge!(dest::FEMesh{S}, src1::FEMesh{S}, src2::FEMesh{S}, others::FEMesh{S}...) where {S}
    merge!(merge!(dest, src1), src2, others...)
end
function Base.merge(x::FEMesh{S}, y::FEMesh{S}, z::FEMesh{S}...) where {S}
    dest = FEMesh(cellshape(x), copy(x.nodes), copy(x.cellsupports))
    merge!(dest, y, z...)
end

"""
    generate_field_meshes(meshes[, order])

Construct consistently numbered field meshes from geometry meshes that share
one node array. Maximum-dimensional cells define the field nodes;
lower-dimensional meshes must match their faces or edges.

Omit `order`, or pass the order already used by every shape, to preserve the
geometry shapes. Pass `Order(1)` to replace them with first-order shapes. The
returned meshes share a compact view of the geometry nodes. Tuple order and
dictionary keys are preserved.
"""
function generate_field_meshes(meshes::Tuple{Vararg{FEMesh}}, order::Union{Nothing, Order}=nothing)
    field_nodes, nodemap = _prepare_field_meshes(meshes, order)
    map(mesh -> _build_field_mesh(mesh, field_nodes, nodemap, order), meshes)
end

function generate_field_meshes(meshes::AbstractDict{K, <: FEMesh}, order::Union{Nothing, Order}=nothing) where {K}
    field_nodes, nodemap = _prepare_field_meshes(values(meshes), order)
    Dict(key => _build_field_mesh(mesh, field_nodes, nodemap, order) for (key, mesh) in meshes)
end

# Return the requested field shape and the positions of its nodes in the geometry connectivity.
_field_interpolation(shape::Shape, ::Nothing) = shape, eachindex(localnodes(shape))
_field_interpolation(shape::Line, ::Order{1}) = Line2(), primarynodes_indices(shape)
_field_interpolation(shape::Quad, ::Order{1}) = Quad4(), primarynodes_indices(shape)
_field_interpolation(shape::Hex, ::Order{1}) = Hex8(), primarynodes_indices(shape)
_field_interpolation(shape::Tri, ::Order{1}) = Tri3(), primarynodes_indices(shape)
_field_interpolation(shape::Tet, ::Order{1}) = Tet4(), primarynodes_indices(shape)
function _field_interpolation(shape::Shape, order::Order)
    typeof(get_order(shape)) === typeof(order) && return shape, eachindex(localnodes(shape))
    throw(ArgumentError("cannot generate an order-$(_order_value(order)) field from $(typeof(shape)) geometry; omit `order` to preserve the geometry shape or pass `Order(1)` to replace it with its first-order shape"))
end

# Preserve cell order and local orientation while translating connectivity to the shared field numbering.
function _build_field_mesh(mesh::FEMesh, field_nodes, nodemap, order)
    geometry_shape = cellshape(mesh)
    field_shape, localindices = _field_interpolation(geometry_shape, order)

    field_supports = map(cells(mesh)) do cell
        indices = supportnodes(mesh, cell)
        field_indices = nodemap[indices[localindices]]
        # Zero marks a geometry node excluded from the field by the largest-dimensional cells.
        any(iszero, field_indices) && throw(ArgumentError("each field node must belong to a maximum-dimensional field cell"))
        field_indices
    end

    FEMesh(field_shape, field_nodes, field_supports)
end

# The largest-dimensional cells define the field. Lower-dimensional meshes may
# only describe their faces or edges and cannot introduce field nodes.
function _prepare_field_meshes(meshes, order)
    isempty(meshes) && throw(ArgumentError("at least one geometry mesh is required"))

    # Connectivity indices are comparable across meshes only when they index the same node array.
    geometry_nodes = first(meshes).nodes
    all(mesh -> mesh.nodes === geometry_nodes, meshes) || throw(ArgumentError("all geometry meshes must share the same node array and numbering"))
    maxdim = maximum(mesh -> get_dimension(cellshape(mesh)), meshes)

    # Pass 1: record every supplied lower-dimensional cell using its complete
    # geometry connectivity, independently of the requested field order.
    unmatched = [Set{Pair{Shape, SVector}}() for _ in 1:maxdim]
    foreach(meshes) do mesh
        shape = cellshape(mesh)
        if get_dimension(shape) != maxdim
            _field_interpolation(shape, order) # reject unsupported shape changes before constructing the numbering
            for cell in cells(mesh)
                indices = supportnodes(mesh, cell)
                push!(unmatched[get_dimension(shape)], shape => sort(indices))
            end
        end
    end

    # Pass 2: largest-dimensional cells select the field nodes; walking their
    # faces and edges removes the lower-dimensional cells recorded above.
    active = falses(length(geometry_nodes))
    foreach(meshes) do mesh
        geometry_shape = cellshape(mesh)
        if get_dimension(geometry_shape) == maxdim
            _, localindices = _field_interpolation(geometry_shape, order)
            for cell in cells(mesh)
                geometry_indices = supportnodes(mesh, cell)
                active[geometry_indices[localindices]] .= true
                any(!isempty, unmatched) && _match_field_subentities!(unmatched, geometry_shape, geometry_indices)
            end
        end
    end
    # Anything left was not an actual face or edge of the largest-dimensional geometry.
    all(isempty, unmatched) || throw(ArgumentError("each lower-dimensional geometry cell must be a subentity of a maximum-dimensional geometry cell"))

    # Dense numbering avoids unused grid entries; the view keeps field coordinates shared with the geometry.
    nodeindices = findall(active)
    nodemap = zeros(Int, length(geometry_nodes))
    nodemap[nodeindices] .= eachindex(nodeindices)
    view(geometry_nodes, nodeindices), nodemap
end

# Walk the cell's faces recursively. Match by shape and complete node set while
# ignoring local orientation.
function _match_field_subentities!(unmatched, shape::Shape, indices)
    delete!(unmatched[get_dimension(shape)], shape => sort(indices))
    get_dimension(shape) == 1 && return

    any(!isempty, @view unmatched[1:get_dimension(shape)-1]) || return

    face_shape = faceshape(shape)
    for face in faces(shape)
        _match_field_subentities!(unmatched, face_shape, indices[face])
    end
end

function _collect_supportnodes(cellsupports)
    nodes = Int[]
    if !isempty(cellsupports)
        sizehint!(nodes, length(cellsupports) * length(first(cellsupports)))
    end
    for indices in cellsupports
        append!(nodes, indices)
    end
    unique!(sort!(nodes))
end

function _refresh_supportnodes!(mesh::FEMesh)
    empty!(mesh.usednodes)
    append!(mesh.usednodes, _collect_supportnodes(mesh.cellsupports))
    mesh
end
