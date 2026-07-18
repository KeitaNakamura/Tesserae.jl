@inline function jet(::Order{1}, shape::Shape, x::Vec)
    grads, vals = gradient(x -> Tensor(value(shape, x)), x, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

"""
    update!(weights::BasisWeightArray{<:Shape}, points::QuadraturePoints, fieldmesh::FEMesh; geometry=fieldmesh, measure=nothing, normal=nothing)

Evaluate the basis of `fieldmesh` using the quadrature rule stored in `points`,
with `geometry` defining the physical mapping. `geometry` defaults to
`fieldmesh`. If they differ, the meshes must describe the same cells in the
same order and orientation.

For full-dimensional cells, `weights` stores basis values and physical
gradients. For boundary cells, it stores basis values. `measure` receives the
physical quadrature measure, and `normal` receives boundary unit normals.
"""
function update!(
        weights::BasisWeightArray{S}, points::QuadraturePoints,
        fieldmesh::FEMesh{S, dim};
        geometry::FEMesh=fieldmesh,
        measure::Union{Nothing, AbstractArray}=nothing,
        normal::Union{Nothing, AbstractArray}=nothing,
    ) where {pdim, dim, S <: Shape{pdim}}
    length(eltype(geometry)) == dim || throw(DimensionMismatch("field and geometry physical dimensions must match"))
    _reference_cell_family(basis(weights)) === _reference_cell_family(cellshape(geometry)) || throw(ArgumentError("field and geometry must use the same reference-cell family"))
    ncells(fieldmesh) == ncells(geometry) || throw(DimensionMismatch("field and geometry must have the same number of cells"))
    set_cellsupports!(getfield(weights, :indices), cellsupports(fieldmesh))
    is_domain = pdim == dim
    mode = pdim == dim ? Val(:domain) : Val(:boundary)
    rule = _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)

    field_shape = cellshape(fieldmesh)
    geometry_shape = cellshape(geometry)
    field_qdata = jet.(Ref(Order(1)), Ref(field_shape), rule.points)
    geometry_qdata = field_shape === geometry_shape ? field_qdata : jet.(Ref(Order(1)), Ref(geometry_shape), rule.points)
    for cell in cells(geometry)
        geometry_indices = supportnodes(geometry, cell)
        x = geometry[geometry_indices]
        for q in eachindex(field_qdata, rule.weights)
            N, dNdξ = field_qdata[q]
            J = sum(x .⊗ last(geometry_qdata[q]))
            _set_quadrature_data!(mode, weights[q,cell], measure, normal, q, cell, N, dNdξ, J, rule.weights[q])
        end
    end
    weights
end

"""
    update!(weights::BasisWeightArray{<:IGABasis}, points::QuadraturePoints, fieldmesh::IGAMesh; geometry=fieldmesh, measure=nothing, normal=nothing)

Evaluate the basis of `fieldmesh` using `geometry` for the physical mapping.
The meshes may have different degrees and control-point numbering, but must
have corresponding patches with the same nonzero knot-span intervals. `measure`
and `normal` have the same roles as in the FEM method.
"""
function update!(
        weights::BasisWeightArray{B}, points::QuadraturePoints,
        fieldmesh::IGAMesh{dim, pdim, T, Degrees};
        geometry::IGAMesh{dim, pdim}=fieldmesh,
        measure::Union{Nothing, AbstractArray}=nothing,
        normal::Union{Nothing, AbstractArray}=nothing,
    ) where {dim, pdim, T, Degrees, B <: IGABasis{pdim, Degrees}}
    is_domain = pdim == dim
    mode = pdim == dim ? Val(:domain) : Val(:boundary)
    rule = _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)
    fieldmesh === geometry && return _update_iga!(mode, weights, rule, fieldmesh, measure, normal)
    check_matching_cell_partitions(fieldmesh, geometry)

    field_supports = cellsupports(getfield(weights, :indices))
    for (field_cell, geometry_cell) in zip(cells(fieldmesh), cells(geometry))
        field_indices = supportnodes(fieldmesh, field_cell)
        @inbounds field_supports[field_cell] = field_indices
        geometry_indices = supportnodes(geometry, geometry_cell)
        field_patch = patches(fieldmesh, field_cell.patch)
        geometry_patch = patches(geometry, geometry_cell.patch)
        x = geometry[geometry_indices]
        for q in eachindex(rule.points, rule.weights)
            field_ξ = span_point(field_patch, field_cell.span, rule.points[q])
            N, dNdξ = iga_basis_values_and_gradients(field_patch, field_cell.span, field_ξ)
            if fieldmesh.weights !== nothing
                N, dNdξ = rational_basis_values_and_gradients(N, dNdξ, fieldmesh.weights[field_indices])
            end
            geometry_ξ = span_point(geometry_patch, geometry_cell.span, rule.points[q])
            geometry_N, geometry_dNdξ = iga_basis_values_and_gradients(geometry_patch, geometry_cell.span, geometry_ξ)
            if geometry.weights !== nothing
                _, geometry_dNdξ = rational_basis_values_and_gradients(geometry_N, geometry_dNdξ, geometry.weights[geometry_indices])
            end
            J = sum(x .⊗ geometry_dNdξ)
            weight = span_weight(geometry_patch, geometry_cell.span, rule.weights[q])
            _set_quadrature_data!(mode, weights[q,field_cell], measure, normal, q, geometry_cell, N, dNdξ, J, weight)
        end
    end
    weights
end

function _update_iga!(mode, weights, rule, mesh, measure, normal)
    field_supports = cellsupports(getfield(weights, :indices))
    for cell in cells(mesh)
        indices = supportnodes(mesh, cell)
        @inbounds field_supports[cell] = indices
        x = mesh[indices]
        patch = patches(mesh, cell.patch)
        for q in eachindex(rule.points, rule.weights)
            ξ = span_point(patch, cell.span, rule.points[q])
            N, dNdξ = iga_basis_values_and_gradients(patch, cell.span, ξ)
            if mesh.weights !== nothing
                N, dNdξ = rational_basis_values_and_gradients(N, dNdξ, mesh.weights[indices])
            end
            J = sum(x .⊗ dNdξ)
            weight = span_weight(patch, cell.span, rule.weights[q])
            _set_quadrature_data!(mode, weights[q,cell], measure, normal, q, cell, N, dNdξ, J, weight)
        end
    end
    weights
end

function _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)
    rule = quadrature_rule(points)
    _check_quadrature_rule(rule, geometry)
    size(weights) == size(points) || throw(DimensionMismatch("basis weights and quadrature points must have the same dimensions"))
    size(points) == (length(rule.points), ncells(geometry)) || throw(DimensionMismatch("quadrature points must have dimensions (rule points, geometry cells)"))
    isnothing(measure) || size(weights) == size(measure) || throw(DimensionMismatch("measure and basis weights must have the same dimensions"))
    is_domain && !isnothing(normal) && throw(ArgumentError("normal is only valid for boundary integration"))
    isnothing(normal) || size(weights) == size(normal) || throw(DimensionMismatch("normal and basis weights must have the same dimensions"))
    if is_domain && !isempty(weights)
        bw = first(weights)
        derivative_order(bw) isa Order{0} && throw(ArgumentError("domain basis weights must store first derivatives"))
        length(eltype(nodal_basis_values(bw, Order(1)))) == length(eltype(geometry)) || throw(DimensionMismatch("basis-gradient and geometry dimensions must match"))
    end
    rule
end

@inline function _set_quadrature_data!(::Val{:domain}, bw, measure, normal, q, cell, N, dNdξ, J, weight)
    set_values!(bw, (N, dNdξ .⊡ Ref(inv(J))))
    if measure !== nothing
        measure[q,cell] = weight * sqrt(det(J'J))
    end
end

@inline function _set_quadrature_data!(::Val{:boundary}, bw, measure, normal, q, cell, N, dNdξ, J, weight)
    set_values!(bw, (N,))
    if measure !== nothing
        measure[q,cell] = weight * sqrt(det(J'J))
    end
    if normal !== nothing
        n = _get_normal(J)
        normal[q,cell] = n / norm(n)
    end
end

_get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
_get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
_get_normal(J::Mat{3,1}) = throw(ArgumentError("normal is not defined for a 3D line element"))
