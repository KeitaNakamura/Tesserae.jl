"""
    IGABasis(degrees)

Basis metadata for tensor-product IGA patches.
"""
struct IGABasis{dim, Degrees <: NTuple{dim, Degree}} <: Basis
    degrees::Degrees
end

degrees(basis::IGABasis) = basis.degrees
basis(mesh::IGAMesh) = IGABasis(degrees(first(patches(mesh))))
nsupportnodes(basis::IGABasis) = prod(degree -> _degree(degree) + 1, degrees(basis))

_generate_supportnodes(::IGABasis, mesh::IGAMesh, dims::Dims{2}) = _generate_cell_supportnodes(mesh, dims)
_generate_supportnodes(::IGABasis, ::IGAMesh, ::Dims) = throw(DimensionMismatch("IGA basis weights must have dimensions (quadrature points, cells)"))

allocate_static_basis_values(::Type{Prop}, basis::IGABasis, dim::Val; derivative::Order) where {Prop <: NamedTuple} =
    allocate_cell_basis_values(Prop, basis, dim; derivative)

"""
    generate_quadrature_rule(basis::IGABasis)
    generate_quadrature_rule(T, basis::IGABasis)

Generate the standard [`QuadratureRule`](@ref) for `basis`, using `p + 1`
Gauss–Legendre points in each parametric direction of degree `p`. `T` may be
`Float16`, `Float32`, or `Float64` and defaults to `Float64`.
"""
generate_quadrature_rule(basis::IGABasis) = generate_quadrature_rule(Float64, basis)
function generate_quadrature_rule(::Type{T}, basis::IGABasis{dim}) where {T, dim}
    _check_quadrature_float(T)
    rules = map(degree -> _gauss_legendre_rule(T, degree), degrees(basis))
    _tensor_product_quadrature_rule(_tensor_product_family(Val(dim)), rules)
end

function _generate_cell_supportnodes(mesh, dims::Dims{2})
    dims[2] == ncells(mesh) || throw(DimensionMismatch("the second basis-weight dimension must equal the number of cells"))
    CellSupportMatrix(map(cell -> supportnodes(mesh, cell), cells(mesh)), dims...)
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
    mode = is_domain ? Val(:domain) : Val(:boundary)
    rule = _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)
    if fieldmesh === geometry
        _update_iga!(Val(true), mode, weights, rule, fieldmesh, geometry, measure, normal)
    else
        check_matching_cell_partitions(fieldmesh, geometry)
        _update_iga!(Val(false), mode, weights, rule, fieldmesh, geometry, measure, normal)
    end
end

function _update_iga!(::Val{same_geometry}, mode, weights, rule, fieldmesh, geometry, measure, normal) where {same_geometry}
    field_supports = cellsupports(supportnodes_storage(weights))
    for (field_cell, geometry_cell) in zip(cells(fieldmesh), cells(geometry))
        field_indices = supportnodes(fieldmesh, field_cell)
        @inbounds field_supports[field_cell] = field_indices
        geometry_indices = same_geometry ? field_indices : supportnodes(geometry, geometry_cell)
        field_patch = patches(fieldmesh, field_cell.patch)
        geometry_patch = same_geometry ? field_patch : patches(geometry, geometry_cell.patch)
        x = geometry[geometry_indices]
        for q in eachindex(rule.points, rule.weights)
            N, dNdξ = iga_span_basis(field_patch, field_cell.span, rule.points[q], fieldmesh.weights, field_indices)
            geometry_dNdξ = same_geometry ? dNdξ : last(iga_span_basis(geometry_patch, geometry_cell.span, rule.points[q], geometry.weights, geometry_indices))
            J = sum(x .⊗ geometry_dNdξ)
            weight = span_weight(geometry_patch, geometry_cell.span, rule.weights[q])
            _set_quadrature_data!(mode, weights[q,field_cell], measure, normal, q, geometry_cell, N, dNdξ, J, weight)
        end
    end
    weights
end
