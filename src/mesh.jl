abstract type AbstractMesh{dim, T, N} <: AbstractArray{Vec{dim, T}, N} end

fillzero!(x::AbstractMesh) = x

"""
    CartesianMesh([T,] h, (xmin, xmax), (ymin, ymax)...)

Construct `CartesianMesh` with the spacing `h`.

# Examples
```jldoctest
julia> CartesianMesh(1.0, (0,3), (1,4))
4×4 CartesianMesh{2, Float64, Vector{Float64}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct CartesianMesh{dim, T, V <: AbstractVector{T}} <: AbstractMesh{dim, T, dim}
    axes::NTuple{dim, V}
    h::T
    h_inv::T
end

Base.size(mesh::CartesianMesh) = map(length, mesh.axes)
Base.IndexStyle(::Type{<: CartesianMesh}) = IndexCartesian()

"""
    spacing(::CartesianMesh)

Return the spacing of the mesh.
"""
spacing(mesh::CartesianMesh) = mesh.h
spacing_inv(mesh::CartesianMesh) = mesh.h_inv

@inline get_xmin(x::CartesianMesh{dim}) where {dim} = @inbounds x[oneunit(CartesianIndex{dim})]
@inline get_xmax(x::CartesianMesh{dim}) where {dim} = @inbounds x[size(x)...]

"""
    volume(::CartesianMesh)

Return the volume of the mesh.
"""
volume(mesh::CartesianMesh) = prod(get_xmax(mesh) - get_xmin(mesh))

function CartesianMesh(axes::Vararg{AbstractRange, dim}) where {dim}
    @assert all(ax->step(ax)==step(first(axes)), axes)
    h = step(first(axes))
    CartesianMesh(axes, h, inv(h))
end

function CartesianMesh(::Type{T}, h::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {T, dim}
    @assert all(x->x[1]<x[2], minmax)
    axes = map(lims->Vector{T}(range(lims...; step=h)), minmax)
    CartesianMesh(axes, T(h), T(inv(h)))
end
CartesianMesh(h::Real, minmax::Tuple{Real, Real}...) = CartesianMesh(Float64, h, minmax...)

"""
    extract(mesh::CartesianMesh, (xmin, xmax), (ymin, ymax)...)

Extract a portion of the `mesh`.
The extracted mesh retains the original origin and spacing.
"""
function extract(mesh::CartesianMesh{dim}, minmax::Vararg{Tuple{Real, Real}, dim}) where {dim}
    @assert all(x->x[1]<x[2], minmax)
    indices = CartesianIndices(ntuple(Val(dim)) do d
        ax = mesh.axes[d]
        xmin, xmax = minmax[d]
        imin =  findlast(x -> (x < xmin) || (x ≈ xmin), ax)
        imax = findfirst(x -> (x > xmax) || (x ≈ xmax), ax)
        imin:imax
    end)
    mesh[indices]
end

axismesh(mesh::CartesianMesh, d::Integer) = CartesianMesh((mesh.axes[d],), spacing(mesh), spacing_inv(mesh))

@generated function Base.getindex(mesh::CartesianMesh{dim}, I::Vararg{Integer, dim}) where {dim}
    quote
        @_inline_meta
        @boundscheck checkbounds(mesh, I...)
        @inbounds Vec(@ntuple $dim d -> mesh.axes[d][I[d]])
    end
end
@generated function Base.getindex(mesh::CartesianMesh{dim}, ranges::Vararg{AbstractUnitRange{Int}, dim}) where {dim}
    quote
        @_inline_meta
        @boundscheck checkbounds(mesh, ranges...)
        @inbounds CartesianMesh((@ntuple $dim d -> mesh.axes[d][ranges[d]]), spacing(mesh), spacing_inv(mesh))
    end
end
@inline function Base.getindex(mesh::CartesianMesh, indices::CartesianIndices)
    @_propagate_inbounds_meta
    mesh[indices.indices...]
end

Base.copy(mesh::CartesianMesh) = CartesianMesh(copy(get_axisarray(mesh)), spacing(mesh), spacing_inv(mesh))

# normalize `x` by `mesh`
@inline function Tensorial.normalize(x::Vec{dim}, mesh::CartesianMesh{dim}) where {dim}
    xmin = get_xmin(mesh)
    h⁻¹ = spacing_inv(mesh)
    (x - xmin) * h⁻¹
end

"""
    isinside(x::Vec, mesh::CartesianMesh)

Check if `x` is inside the `mesh`.
"""
@inline function isinside(x::Vec{dim}, mesh::CartesianMesh{dim}) where {dim}
    ξ = Tuple(normalize(x, mesh))
    isinside(ξ, size(mesh))
end
@inline function isinside(ξ::NTuple{dim}, dims::Dims{dim}) where {dim}
    !isnothing(_findcell(ξ, dims))
end

"""
    neighboringnodes(x::Vec, r::Real, mesh::CartesianMesh)

Return `CartesianIndices` for neighboring nodes around `x`.
`r` denotes the range for searching area. In 1D, for example, the range `a`
becomes ` x-r*h ≤ a < x+r*h` where `h` is `spacing(mesh)`.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,5))
6-element CartesianMesh{1, Float64, Vector{Float64}}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> neighboringnodes(Vec(1.5), 1, mesh)
CartesianIndices((2:3,))

julia> neighboringnodes(Vec(1.5), 3, mesh)
CartesianIndices((1:5,))
```
"""
@inline function neighboringnodes(x::Vec, r::Real, mesh::CartesianMesh{dim, T}) where {dim, T}
    ξ = Tuple(normalize(x, mesh))
    dims = size(mesh)
    isinside(ξ, dims) || return EmptyCartesianIndices(Val(dim))
    start = @. unsafe_trunc(Int, floor(ξ - r)) + 2
    stop  = @. unsafe_trunc(Int, floor(ξ + r)) + 1
    imin = Tuple(@. max(start, 1))
    imax = Tuple(@. min(stop, dims))
    CartesianIndices(UnitRange.(imin, imax))
end
@inline EmptyCartesianIndices(::Val{dim}) where {dim} = CartesianIndices(nfill(1:0, Val(dim)))

"""
    findcell(x::Vec, mesh::CartesianMesh)

Return the cell index where `x` is located.

# Examples
```jldoctest
julia> mesh = CartesianMesh(1, (0,5), (0,5))
6×6 CartesianMesh{2, Float64, Vector{Float64}}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> findcell(Vec(1.5, 1.5), mesh)
CartesianIndex(2, 2)
```
"""
@inline function findcell(x::Vec, mesh::CartesianMesh{dim, T}) where {dim, T}
    ξ = Tuple(normalize(x, mesh))
    _findcell(ξ, size(mesh))
end

@generated function _findcell(ξ::NTuple{dim, T}, gridsize::Dims{dim}) where {dim, T}
    quote
        @_inline_meta
        index = map(floor, ξ)
        isinside = @nall $dim d -> T(0) ≤ index[d] ≤ T(gridsize[d]-2)
        ifelse(isinside, CartesianIndex(@. unsafe_trunc(Int, index) + 1), nothing)
    end
end

struct UnstructuredMesh{S <: Shape, dim, T, L} <: AbstractMesh{dim, T, 1}
    shape::S
    nodes::Vector{Vec{dim, T}}
    cellnodeindices::Vector{SVector{L, Int}}
end

Base.size(mesh::UnstructuredMesh) = size(mesh.nodes)
Base.IndexStyle(::Type{<: UnstructuredMesh}) = IndexLinear()

@inline function Base.getindex(mesh::UnstructuredMesh, i::Int)
    @_propagate_inbounds_meta
    mesh.nodes[i]
end
@inline function Base.setindex!(mesh::UnstructuredMesh, x, i::Int)
    @_propagate_inbounds_meta
    mesh.nodes[i] = x
end

cellshape(mesh::UnstructuredMesh) = mesh.shape
ncells(mesh::UnstructuredMesh) = length(mesh.cellnodeindices)

@inline cellnodeindices(mesh::UnstructuredMesh, c::Int) = (@_propagate_inbounds_meta; mesh.cellnodeindices[c])

UnstructuredMesh(mesh::CartesianMesh) = UnstructuredMesh(default_cellshape(mesh), mesh)
default_cellshape(::CartesianMesh{1}) = Line2()
default_cellshape(::CartesianMesh{2}) = Quad4()
default_cellshape(::CartesianMesh{3}) = Hex8()

function UnstructuredMesh(shape::Shape{dim}, mesh::CartesianMesh{dim}) where {dim}
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

    UnstructuredMesh(shape, mesh′[findall(>(0), nodeindices)], vec(connecitivies))
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
    end...)
end

function extract(mesh::UnstructuredMesh, nodeindices::AbstractVector{Int})
    shape = cellshape(mesh)
    cells = SVector{nlocalnodes(shape), Int}[]
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        if all(in.(indices, Ref(nodeindices)))
            push!(cells, indices)
        end
    end
    UnstructuredMesh(shape, mesh.nodes, cells)
end

function extract_face(mesh::UnstructuredMesh, nodeindices::AbstractVector{Int})
    shape = faceshape(cellshape(mesh))
    cells = SVector{nlocalnodes(shape), Int}[]
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        for conn in faces(cellshape(mesh))
            faceindices = indices[conn]
            if all(in.(faceindices, Ref(nodeindices)))
                push!(cells, faceindices)
            end
        end
    end
    UnstructuredMesh(shape, mesh.nodes, unique(cells))
end

function Base.merge!(dest::UnstructuredMesh{S}, src::UnstructuredMesh{S}) where {S}
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

    sortedinds_src = map(inds -> sort(nodemap[inds]), src.cellnodeindices)
    sortedinds_dst = map(sort, dest.cellnodeindices)
    for c in 1:ncells(src)
        inds_src = sortedinds_src[c]
        if all(inds_dest -> inds_src !== inds_dest, sortedinds_dst)
            push!(dest.cellnodeindices, nodemap[cellnodeindices(src, c)])
        end
    end
    dest
end
function Base.merge!(dest::UnstructuredMesh{S}, src1::UnstructuredMesh{S}, src2::UnstructuredMesh{S}, others::UnstructuredMesh{S}...) where {S}
    merge!(merge!(dest, src1), src2, others...)
end
function Base.merge(x::UnstructuredMesh{S}, y::UnstructuredMesh{S}, z::UnstructuredMesh{S}...) where {S}
    dest = UnstructuredMesh(cellshape(x), copy(x.nodes), copy(x.cellnodeindices))
    merge!(dest, y, z...)
end
