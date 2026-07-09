@inline function jet(::Order{1}, shape::Shape, x::Vec)
    grads, vals = gradient(x -> Tensor(value(shape, x)), x, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

function feupdate!(
        weights::AbstractArray{<: BasisWeight{S}}, mesh::FEMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        measure::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(cellshape(mesh)),
    ) where {dim, S <: Shape{dim}}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    qdata = jet.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    _feupdate!(Val(:domain), weights, mesh, nodes, measure, nothing, qdata, qwts)
end

function feupdate!(
        weights::AbstractArray{<: BasisWeight{<: IGABasis}}, mesh::IGAMesh{dim, dim, T},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        measure::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(igabasis(mesh)),
    ) where {dim, T}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    _feupdate!(Val(:domain), weights, mesh, nodes, measure, nothing, qpts, qwts)
end

@inline _basis_values_and_gradients(mesh::FEMesh, qdata, cell, indices, p) = qdata[p]
@inline _quadrature_weight(mesh::FEMesh, qdata, qwts, cell, p) = qwts[p]
@inline function _basis_values_and_gradients(mesh::IGAMesh, qdata, cell, indices, p)
    patch = patches(mesh, cell.patch)
    ξ = span_point(patch, cell.span, qdata[p])
    N, dN = iga_basis_values_and_gradients(patch, cell.span, ξ)
    _rationalize_basis(N, dN, mesh.weights, indices)
end
@inline _rationalize_basis(N, dN, ::Nothing, indices) = N, dN
@inline _rationalize_basis(N, dN, weights::AbstractVector, indices) = rational_basis_values_and_gradients(N, dN, weights[indices])
@inline function _quadrature_weight(mesh::IGAMesh, qdata, qwts, cell, p)
    patch = patches(mesh, cell.patch)
    span_weight(patch, cell.span, qwts[p])
end

function _feupdate!(mode, weights, mesh::AbstractMesh, nodes, measure, normal, qdata, qwts)
    @assert length(qdata) == length(qwts)
    @assert size(mesh) == size(nodes)
    @assert size(weights) == (length(qdata), ncells(mesh))
    _check_feupdate_outputs(mode, weights, measure, normal)
    for cell in cells(mesh)
        indices = supportnodes(mesh, cell)
        x = nodes[indices]
        for p in eachindex(qdata, qwts)
            N, dNdξ = _basis_values_and_gradients(mesh, qdata, cell, indices, p)
            J = sum(x .⊗ dNdξ)
            qwt = _quadrature_weight(mesh, qdata, qwts, cell, p)
            bw = weights[p,cell]
            _set_feupdate_values!(mode, bw, N, dNdξ, J)
            supportnodes_storage(bw)[] = indices
            _set_feupdate_outputs!(mode, measure, normal, p, cell, qwt, J)
        end
    end
end

function _check_feupdate_outputs(::Val{:domain}, weights, measure, normal)
    @assert isnothing(measure) || size(weights) == size(measure)
    @assert isnothing(normal)
end
function _check_feupdate_outputs(::Val{:boundary}, weights, measure, normal)
    @assert isnothing(measure) || size(weights) == size(measure)
    @assert isnothing(normal) || size(weights) == size(normal)
end

# This is dL, dA, or dV depending on the parametric dimension of the element.
@inline _jacobian_measure(qwt, J) = qwt * sqrt(det(J'J))

# Domain cells have a square Jacobian, so gradients can be mapped to physical coordinates.
@inline _set_feupdate_values!(::Val{:domain}, bw, N, dNdξ, J) = set_values!(bw, (N, dNdξ .⊡ Ref(inv(J))))
# Boundary cells only need basis values for line/surface integration.
@inline _set_feupdate_values!(::Val{:boundary}, bw, N, dNdξ, J) = set_values!(bw, (N,))

_get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
_get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
_get_normal(J::Mat{3,1}) = throw(ArgumentError("normal is not defined for a 3D line element"))

@inline function _set_feupdate_outputs!(::Val{:domain}, measure, normal, p, cell, qwt, J)
    if measure !== nothing
        measure[p,cell] = _jacobian_measure(qwt, J)
    end
end
@inline function _set_feupdate_outputs!(::Val{:boundary}, measure, normal, p, cell, qwt, J)
    if measure !== nothing
        measure[p,cell] = _jacobian_measure(qwt, J)
    end
    if normal !== nothing
        n = _get_normal(J)
        n_norm = norm(n)
        normal[p,cell] = n / n_norm
    end
end

function feupdate!(
        weights::AbstractArray{<: BasisWeight{S}}, mesh::FEMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        measure::Union{Nothing, AbstractArray} = nothing, normal::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(cellshape(mesh)),
    ) where {S <: Shape, dim}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    qdata = jet.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    _feupdate!(Val(:boundary), weights, mesh, nodes, measure, normal, qdata, qwts)
end

function feupdate!(
        weights::AbstractArray{<: BasisWeight{<: IGABasis}}, mesh::IGAMesh{dim, pdim, T},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        measure::Union{Nothing, AbstractArray} = nothing, normal::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(igabasis(mesh)),
    ) where {dim, T, pdim}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    _feupdate!(Val(:boundary), weights, mesh, nodes, measure, normal, qpts, qwts)
end
