@inline function jet(::Order{1}, shape::Shape, x::Vec)
    grads, vals = gradient(x -> Tensor(value(shape, x)), x, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

"""
    update!(weights::BasisWeightArray{<:Shape}, points::QuadraturePoints, geometry::FEMesh; measure=nothing, normal=nothing)

Evaluate the FEM field basis stored by `weights` at the reference points in
`quadrature_rule(points)`. `geometry` supplies the mapping to physical space.
The field and geometry may use different shapes from the same reference-cell
family, provided that they describe the same cells in the same order and
orientation.

For full-dimensional cells, `weights` stores basis values and physical
gradients. For boundary cells, it stores basis values. `measure` receives the
physical quadrature measure, and `normal` receives boundary unit normals.
"""
function update!(
        weights::BasisWeightArray{S}, points::QuadraturePoints,
        geometry::FEMesh{<: Shape{pdim}, dim};
        measure::Union{Nothing, AbstractArray}=nothing,
        normal::Union{Nothing, AbstractArray}=nothing,
    ) where {pdim, dim, S <: Shape{pdim}}
    _reference_cell_family(basis(weights)) === _reference_cell_family(cellshape(geometry)) || throw(ArgumentError("field and geometry must use the same reference-cell family"))
    is_domain = pdim == dim
    mode = pdim == dim ? Val(:domain) : Val(:boundary)
    rule = _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)

    field_shape = basis(weights)
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
    update!(weights::BasisWeightArray{<:IGABasis}, points::QuadraturePoints, geometry::IGAMesh; measure=nothing, normal=nothing)

Evaluate the IGA basis of `geometry` at the reference points in
`quadrature_rule(points)` and store the result in `weights`. `measure` and
`normal` have the same roles as in the FEM method.
"""
function update!(
        weights::BasisWeightArray{B}, points::QuadraturePoints,
        geometry::IGAMesh{dim, pdim};
        measure::Union{Nothing, AbstractArray}=nothing,
        normal::Union{Nothing, AbstractArray}=nothing,
    ) where {dim, pdim, B <: IGABasis{pdim}}
    degrees(basis(weights)) == degrees(igabasis(geometry)) || throw(ArgumentError("IGA field and geometry degrees must match"))
    is_domain = pdim == dim
    mode = pdim == dim ? Val(:domain) : Val(:boundary)
    rule = _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)

    for cell in cells(geometry)
        indices = supportnodes(geometry, cell)
        supportnodes(weights[1,cell]) == indices || throw(ArgumentError("IGA basis weights and geometry must use the same cell connectivity"))
        x = geometry[indices]
        patch = patches(geometry, cell.patch)
        for q in eachindex(rule.points, rule.weights)
            ξ = span_point(patch, cell.span, rule.points[q])
            N, dNdξ = iga_basis_values_and_gradients(patch, cell.span, ξ)
            if geometry.weights !== nothing
                N, dNdξ = rational_basis_values_and_gradients(N, dNdξ, geometry.weights[indices])
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
