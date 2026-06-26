@inline function jet(::Order{1}, shape::Shape, x::Vec)
    grads, vals = gradient(x -> Tensor(value(shape, x)), x, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

function feupdate!(
        weights::AbstractArray{<: BasisWeight{S}}, mesh::UnstructuredMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        volume::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(cellshape(mesh)),
    ) where {dim, S <: Shape{dim}}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    qdata = jet.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    _feupdate_volume!(weights, mesh, nodes, volume, qdata, qwts)
end

@inline _basis_values_and_gradients(mesh::UnstructuredMesh, qdata, cell, p) = qdata[p]
@inline _quadrature_weight(mesh::UnstructuredMesh, qdata, qwts, cell, p) = qwts[p]

function _feupdate_volume!(weights, mesh::AbstractMesh, nodes, volume, qdata, qwts)
    @assert length(qdata) == length(qwts)
    @assert size(mesh) == size(nodes)
    @assert size(weights) == (length(qdata), ncells(mesh))
    @assert isnothing(volume) || size(weights) == size(volume)
    for cell in cells(mesh)
        indices = supportnodes(mesh, cell)
        x = nodes[indices]
        for p in eachindex(qdata, qwts)
            N, dNdξ = _basis_values_and_gradients(mesh, qdata, cell, p)
            J = sum(x .⊗ dNdξ)
            set_values!(weights[p,cell], (N, dNdξ .⊡ Ref(inv(J))))
            supportnodes_storage(weights[p,cell])[] = indices
            if volume !== nothing
                volume[p,cell] = _quadrature_weight(mesh, qdata, qwts, cell, p) * det(J)
            end
        end
    end
end

_get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
_get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
function feupdate!(
        weights::AbstractArray{<: BasisWeight{S}}, mesh::UnstructuredMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        area::Union{Nothing, AbstractArray} = nothing, normal::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(cellshape(mesh)),
    ) where {S <: Shape, dim}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    qdata = jet.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    _feupdate_area!(weights, mesh, nodes, area, normal, qdata, qwts)
end

function _feupdate_area!(weights, mesh::AbstractMesh, nodes, area, normal, qdata, qwts)
    @assert length(qdata) == length(qwts)
    @assert size(mesh) == size(nodes)
    @assert size(weights) == (length(qdata), ncells(mesh))
    @assert isnothing(area) || size(weights) == size(area)
    @assert isnothing(normal) || size(weights) == size(normal)
    for cell in cells(mesh)
        indices = supportnodes(mesh, cell)
        x = nodes[indices]
        for p in eachindex(qdata, qwts)
            N, dNdξ = _basis_values_and_gradients(mesh, qdata, cell, p)
            J = sum(x .⊗ dNdξ)
            n = _get_normal(J)
            n_norm = norm(n)
            set_values!(weights[p,cell], (N,))
            supportnodes_storage(weights[p,cell])[] = indices
            if area !== nothing
                area[p,cell] = _quadrature_weight(mesh, qdata, qwts, cell, p) * n_norm
            end
            if normal !== nothing
                normal[p,cell] = n / n_norm
            end
        end
    end
end
