# FEM basis weights

allocate_static_basis_values(::Type{Prop}, shape::Shape, dim::Val; derivative::Order) where {Prop <: NamedTuple} =
    allocate_cell_basis_values(Prop, shape, dim; derivative)

@inline basis(mesh::FEMesh) = cellshape(mesh)

_generate_supportnodes(::Shape, mesh::FEMesh, dims::Dims{2}) = CellSupportMatrix(cellsupports(mesh), dims...)
_generate_supportnodes(::Shape, ::FEMesh, ::Dims) = throw(DimensionMismatch("FEM basis weights must have dimensions (quadrature points, cells)"))

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
    is_domain = pdim == dim
    mode = is_domain ? Val(:domain) : Val(:boundary)
    rule = _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)
    set_cellsupports!(getfield(weights, :indices), cellsupports(fieldmesh))

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
