@inline function Base.values(::Order{1}, shape::Shape, x::Vec)
    grads, vals = gradient(x -> Tensor(value(shape, x)), x, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

function feupdate!(
        weights::AbstractArray{<: InterpolationWeight{S}}, mesh::UnstructuredMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        volume::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(cellshape(mesh)),
    ) where {dim, S <: Shape{dim}}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    @assert length(qpts) == length(qwts)
    @assert size(mesh) == size(nodes)
    @assert size(weights) == (length(qpts), ncells(mesh))
    @assert isnothing(volume) || size(weights) == size(volume)
    valgrads = values.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        x = nodes[indices]
        for p in eachindex(qpts, qwts)
            N, dNdξ = valgrads[p]
            J = sum(x .⊗ dNdξ)
            set_values!(weights[p,c], (N, dNdξ .⊡ Ref(inv(J))))
            neighboringnodes_storage(weights[p,c])[] = indices
            if volume !== nothing
                volume[p,c] = qwts[p] * det(J)
            end
        end
    end
end

_get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
_get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
function feupdate!(
        weights::AbstractArray{<: InterpolationWeight{S}}, mesh::UnstructuredMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        area::Union{Nothing, AbstractArray} = nothing, normal::Union{Nothing, AbstractArray} = nothing,
        quadrature_rule::QuadratureRule = quadrature_rule(cellshape(mesh)),
    ) where {S <: Shape, dim}
    qpts, qwts = quadrature_rule.points, quadrature_rule.weights
    @assert length(qpts) == length(qwts)
    @assert size(mesh) == size(nodes)
    @assert size(weights) == (length(qpts), ncells(mesh))
    @assert isnothing(area) || size(weights) == size(area)
    @assert isnothing(normal) || size(weights) == size(normal)
    valgrads = values.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        x = nodes[indices]
        for p in eachindex(qpts, qwts)
            N, dNdξ = valgrads[p]
            J = sum(x .⊗ dNdξ)
            n = _get_normal(J)
            n_norm = norm(n)
            set_values!(weights[p,c], (N,))
            neighboringnodes_storage(weights[p,c])[] = indices
            if area !== nothing
                area[p,c] = qwts[p] * n_norm
            end
            if normal !== nothing
                normal[p,c] = n / n_norm
            end
        end
    end
end
