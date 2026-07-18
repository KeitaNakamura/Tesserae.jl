_degree(::Degree{p}) where {p} = p

"""
    IGAPatch(degrees, knot_vectors, controlpoint_ids)

Define a tensor-product IGA patch.
"""
struct IGAPatch{pdim, T, Degrees <: NTuple{pdim, Degree}}
    degrees::Degrees
    knot_vectors::NTuple{pdim, Vector{T}}
    controlpoint_ids::Array{Int, pdim}
    function IGAPatch{pdim, T, Degrees}(degrees::Degrees, knot_vectors::NTuple{pdim, Vector{T}}, controlpoint_ids::Array{Int, pdim}) where {pdim, T, Degrees <: NTuple{pdim, Degree}}
        _check_iga_patch(degrees, knot_vectors, controlpoint_ids)
        new{pdim, T, Degrees}(degrees, knot_vectors, controlpoint_ids)
    end
end

function IGAPatch(degrees::Degrees, knot_vectors::NTuple{pdim, Vector{T}}, controlpoint_ids::Array{Int, pdim}) where {pdim, T, Degrees <: NTuple{pdim, Degree}}
    IGAPatch{pdim, T, Degrees}(degrees, knot_vectors, controlpoint_ids)
end

degrees(patch::IGAPatch) = patch.degrees
degrees(patch::IGAPatch, i::Integer) = degrees(patch)[i]

function _check_iga_patch(degrees, knot_vectors, controlpoint_ids)
    for d in eachindex(degrees)
        p = _degree(degrees[d])
        knot_vector = knot_vectors[d]
        p < 0 && throw(ArgumentError("degrees must be non-negative"))
        length(knot_vector) < p + 2 && throw(ArgumentError("knot vector length must be at least degree + 2"))
        issorted(knot_vector) || throw(ArgumentError("knot vectors must be sorted"))
        nbasis = length(knot_vector) - p - 1
        size(controlpoint_ids, d) == nbasis || throw(ArgumentError("controlpoint_ids size must match knot vector and degree"))
    end
    nothing
end

function _active_span_range(knot_vector, degree)
    p = _degree(degree)
    (p+1):(length(knot_vector)-p-1)
end

@inline _has_positive_span(knot_vector, span::Integer) = knot_vector[span] < knot_vector[span+1]
@inline function _has_positive_span(patch::IGAPatch{pdim}, span::CartesianIndex{pdim}) where {pdim}
    all(d -> _has_positive_span(patch.knot_vectors[d], span[d]), 1:pdim)
end
_span_ranges(patch::IGAPatch) = map(_active_span_range, patch.knot_vectors, degrees(patch))
_span_count(knot_vector, degree) = count(i -> _has_positive_span(knot_vector, i), _active_span_range(knot_vector, degree))

# Repeated knots are continuity markers; only positive knot intervals become cells.
_span_indices(patch::IGAPatch) = Iterators.filter(span -> _has_positive_span(patch, span), CartesianIndices(_span_ranges(patch)))

"""
    IGAMesh(patches, controlpoints[, weights])
    IGAMesh(mesh::CartesianMesh; degree, weights=nothing)

Define an IGA mesh from patches and control points.
"""
struct IGAMesh{dim, pdim, T, Degrees <: NTuple{pdim, Degree}} <: AbstractMesh{dim, T, 1}
    patches::Vector{IGAPatch{pdim, T, Degrees}}
    controlpoints::Vector{Vec{dim, T}}
    weights::Union{Nothing, Vector{T}}
    used_controlpoint_ids::Vector{Int}
    function IGAMesh{dim, pdim, T, Degrees}(patches::Vector{IGAPatch{pdim, T, Degrees}}, controlpoints::Vector{Vec{dim, T}}, weights::Union{Nothing, Vector{T}}) where {dim, pdim, T, Degrees <: NTuple{pdim, Degree}}
        _check_iga_mesh(patches, controlpoints, weights)
        new{dim, pdim, T, Degrees}(patches, controlpoints, weights, _collect_used_controlpoint_ids(patches))
    end
end

function _check_iga_mesh(patches, controlpoints, weights)
    isempty(patches) && throw(ArgumentError("patches must not be empty"))
    _check_controlpoint_weights(weights, controlpoints)
    for patch in patches
        ids = patch.controlpoint_ids
        minimum(ids) < 1 && throw(ArgumentError("controlpoint ids must be positive"))
        maximum(ids) > length(controlpoints) && throw(ArgumentError("controlpoint ids must not exceed the number of control points"))
    end
    nothing
end
_check_controlpoint_weights(::Nothing, controlpoints) = nothing
function _check_controlpoint_weights(weights::AbstractVector, controlpoints)
    length(weights) == length(controlpoints) || throw(ArgumentError("weights length must match controlpoints length"))
    nothing
end

function _collect_used_controlpoint_ids(patches::AbstractVector{<: IGAPatch})
    nodes = Int[]
    for patch in patches
        append!(nodes, patch.controlpoint_ids)
    end
    unique!(sort!(nodes))
end

function IGAMesh(patches::Vector{IGAPatch{pdim, T, Degrees}}, controlpoints::Vector{Vec{dim, T}}, weights=nothing) where {dim, T, pdim, Degrees <: NTuple{pdim, Degree}}
    IGAMesh{dim, pdim, T, Degrees}(patches, controlpoints, _controlpoint_weights(T, weights))
end

function IGAMesh(mesh::CartesianMesh{dim, T}; degree::Degree, weights=nothing) where {dim, T}
    patch_degrees = _uniform_degrees(degree, Val(dim))
    patch, controlpoints = _patch_from_cartesian_mesh(mesh, patch_degrees)
    IGAMesh([patch], controlpoints, _controlpoint_weights(T, weights))
end

"""
    IGAMesh(net::NURBS.ControlNet; merge=false, atol=nothing, rtol=nothing)
    IGAMesh(nets::AbstractVector{<: NURBS.ControlNet}; merge=false, atol=nothing, rtol=nothing)

Create an IGA mesh from one or more tensor-product NURBS control nets. If
`merge` is true, matching control points with matching weights share a global
control-point id. `atol` and `rtol`, when given, are passed to `isapprox` for
matching.
"""
function IGAMesh(net::NURBS.ControlNet{dim, pdim}; merge::Bool=false, atol=nothing, rtol=nothing) where {dim, pdim}
    IGAMesh([net]; merge, atol, rtol)
end

function IGAMesh(nets::AbstractVector{<: NURBS.ControlNet{dim, pdim, T}}; merge::Bool=false, atol=nothing, rtol=nothing) where {dim, pdim, T}
    isempty(nets) && throw(ArgumentError("control nets must not be empty"))

    patch_degrees = degrees(first(nets))
    patches = IGAPatch{pdim, T, typeof(patch_degrees)}[]
    controlpoints = Vec{dim, T}[]
    weights = T[]
    for net in nets
        degrees(net) == patch_degrees || throw(ArgumentError("control net degrees must match"))
        ids = register_controlpoints_and_weights!(controlpoints, weights, net; merge, atol, rtol)
        push!(patches, IGAPatch(patch_degrees, knot_vectors(net), ids))
    end
    IGAMesh(patches, controlpoints, weights)
end

degrees(net::NURBS.ControlNet) = map(axis -> Degree(axis.degree), net.axes)
knot_vectors(net::NURBS.ControlNet) = map(axis -> copy(axis.knot_vector), net.axes)

function register_controlpoints_and_weights!(controlpoints, weights, net::NURBS.ControlNet; merge::Bool, atol, rtol)
    controlpoint_ids = Array{Int}(undef, size(net.points))
    for I in CartesianIndices(net.points)
        id = merge ? matching_controlpoint(controlpoints, weights, net.points[I], net.weights[I], atol, rtol) : 0
        if iszero(id)
            push!(controlpoints, net.points[I])
            push!(weights, net.weights[I])
            id = length(controlpoints)
        end
        controlpoint_ids[I] = id
    end
    controlpoint_ids
end

function matching_controlpoint(controlpoints, weights, point, weight, atol, rtol)
    for i in eachindex(controlpoints)
        approx_equal(controlpoints[i], point, atol, rtol) || continue
        approx_equal(weights[i], weight, atol, rtol) || continue
        return i
    end
    0
end

function approx_equal(a, b, atol, rtol)
    isnothing(atol) && isnothing(rtol) && return isapprox(a, b)
    isnothing(atol) && return isapprox(a, b; rtol)
    isnothing(rtol) && return isapprox(a, b; atol)
    isapprox(a, b; atol, rtol)
end

_controlpoint_weights(::Type{T}, ::Nothing) where {T} = nothing
_controlpoint_weights(::Type{T}, weights::AbstractVector) where {T} = Vector{T}(weights)

function _uniform_degrees(degree::Degree, ::Val{dim}) where {dim}
    p = _degree(degree)
    p < 1 && throw(ArgumentError("degree must be positive"))
    nfill(degree, Val(dim))
end

function _patch_from_cartesian_mesh(mesh::CartesianMesh, degrees)
    knot_vectors = map(_open_knot_vector_from_axis, mesh.axes, degrees)
    controlpoint_axes = map(_greville_abscissae, knot_vectors, degrees)
    controlpoint_ids, controlpoints = _tensor_product_controlpoints(controlpoint_axes)
    IGAPatch(degrees, knot_vectors, controlpoint_ids), controlpoints
end

function _open_knot_vector_from_axis(axis::AbstractVector{T}, degree::Degree) where {T}
    p = _degree(degree)
    length(axis) < 2 && throw(ArgumentError("CartesianMesh axis must contain at least two points"))
    vcat(fill(first(axis), p+1), axis[2:end-1], fill(last(axis), p+1))
end

function _greville_abscissae(knot_vector::AbstractVector{T}, degree::Degree) where {T}
    p = _degree(degree)
    nbasis = length(knot_vector) - p - 1
    map(1:nbasis) do i
        sum(j -> knot_vector[i+j], 1:p) / p
    end
end

function _tensor_product_controlpoints(axes::NTuple{dim}) where {dim}
    controlpoint_ids = LinearIndices(map(length, axes))
    controlpoints = map(CartesianIndices(controlpoint_ids)) do I
        Vec(map(getindex, axes, Tuple(I)))
    end
    Array(controlpoint_ids), vec(controlpoints)
end

patches(mesh::IGAMesh) = mesh.patches
patches(mesh::IGAMesh, i::Integer) = patches(mesh)[i]

Base.size(mesh::IGAMesh) = size(mesh.controlpoints)
Base.IndexStyle(::Type{<: IGAMesh}) = IndexLinear()

@inline function Base.getindex(mesh::IGAMesh, i::Int)
    @_propagate_inbounds_meta
    mesh.controlpoints[i]
end

@inline function Base.setindex!(mesh::IGAMesh, x, i::Int)
    @_propagate_inbounds_meta
    mesh.controlpoints[i] = x
end

"""
    boundaries(mesh, patch_id)
    boundaries(mesh, patch_id, direction, side)

Extract boundary IGA meshes from a parent IGA mesh. The boundary patch keeps the
parent control-point ids, so boundary assembly targets the same global DOFs.
"""
function boundaries(mesh::IGAMesh{dim, 2}, patch_id::Integer) where {dim}
    tuple(
        boundaries(mesh, patch_id, 1, -1),
        boundaries(mesh, patch_id, 1, +1),
        boundaries(mesh, patch_id, 2, -1),
        boundaries(mesh, patch_id, 2, +1),
    )
end

function boundaries(mesh::IGAMesh{dim, 3}, patch_id::Integer) where {dim}
    tuple(
        boundaries(mesh, patch_id, 1, -1),
        boundaries(mesh, patch_id, 1, +1),
        boundaries(mesh, patch_id, 2, -1),
        boundaries(mesh, patch_id, 2, +1),
        boundaries(mesh, patch_id, 3, -1),
        boundaries(mesh, patch_id, 3, +1),
    )
end

function boundaries(mesh::IGAMesh{dim, pdim, T}, patch_id::Integer, direction::Int, side::Integer) where {dim, pdim, T}
    2 ≤ pdim ≤ 3 || throw(ArgumentError("boundaries supports surfaces and volumes"))
    1 ≤ patch_id ≤ length(mesh.patches) || throw(ArgumentError("patch_id must be a valid patch index"))
    1 ≤ direction ≤ pdim || throw(ArgumentError("direction must be between 1 and the parametric dimension"))
    (side == -1 || side == 1) || throw(ArgumentError("side must be -1 or +1"))

    parent_patch = mesh.patches[patch_id]
    fixed = side == -1 ? firstindex(parent_patch.controlpoint_ids, direction) : lastindex(parent_patch.controlpoint_ids, direction)
    indices = ntuple(d -> d == direction ? fixed : (:), Val(pdim))
    boundary_patch = IGAPatch(
        dropat(parent_patch.degrees, direction),
        dropat(parent_patch.knot_vectors, direction),
        parent_patch.controlpoint_ids[indices...],
    )
    IGAMesh([boundary_patch], mesh.controlpoints, mesh.weights)
end

# IGACell keeps the array id and the tensor-product knot span together.
struct IGACell{pdim}
    id::Int
    patch::Int
    span::CartesianIndex{pdim}
end
@inline Base.to_index(cell::IGACell) = cell.id

ncells(mesh::IGAMesh) = sum(_ncells, patches(mesh))
_ncells(patch::IGAPatch) = prod(map(_span_count, patch.knot_vectors, degrees(patch)))

function cells(mesh::IGAMesh)
    spans = Iterators.flatten(_patch_cells(p, patch) for (p, patch) in pairs(patches(mesh)))
    (IGACell(id, patch, span) for (id, (patch, span)) in enumerate(spans))
end
_patch_cells(p, patch) = ((p, span) for span in _span_indices(patch))

# Check that both meshes enumerate the same nonzero knot-span cells in the same order.
function check_matching_cell_partitions(mesh1::IGAMesh, mesh2::IGAMesh)
    length(patches(mesh1)) == length(patches(mesh2)) || throw(DimensionMismatch("IGA meshes must have the same number of patches"))
    for (patch1, patch2) in zip(patches(mesh1), patches(mesh2))
        for d in eachindex(degrees(patch1))
            knots1 = patch1.knot_vectors[d]
            knots2 = patch2.knot_vectors[d]
            spans1 = Iterators.filter(i -> _has_positive_span(knots1, i), _active_span_range(knots1, degrees(patch1, d)))
            spans2 = Iterators.filter(i -> _has_positive_span(knots2, i), _active_span_range(knots2, degrees(patch2, d)))
            _span_count(knots1, degrees(patch1, d)) == _span_count(knots2, degrees(patch2, d)) || throw(DimensionMismatch("IGA meshes must have the same number of nonzero knot spans"))
            for (span1, span2) in zip(spans1, spans2)
                knots1[span1] == knots2[span2] && knots1[span1+1] == knots2[span2+1] || throw(ArgumentError("IGA meshes must have the same nonzero knot spans"))
            end
        end
    end
    nothing
end

"""
    supportnodes(mesh::IGAMesh)
    supportnodes(mesh::IGAMesh, cell::IGACell)

Return the sorted control-point indices used by `mesh`, or the local support
control-point indices of `cell`.
"""
@inline supportnodes(mesh::IGAMesh) = mesh.used_controlpoint_ids
@generated function supportnodes(mesh::IGAMesh{dim, pdim, T, Degrees}, cell::IGACell{pdim}) where {dim, pdim, T, Degrees <: NTuple{pdim, Degree}}
    p = map(degree -> degree.parameters[1], fieldtypes(Degrees))
    support_dims = p .+ 1
    ids = map(CartesianIndices(support_dims)) do I
        indices = :(Tuple(cell.span) .- $p .+ $(Tuple(I) .- 1))
        :(patch.controlpoint_ids[$(indices)...])
    end
    quote
        @_inline_meta
        patch = patches(mesh, cell.patch)
        SVector{$(prod(support_dims)), Int}($(ids...))
    end
end
