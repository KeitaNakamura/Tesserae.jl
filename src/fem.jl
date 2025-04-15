@inline function Base.values(::Order{1}, shape::Shape, x::Vec)
    grads, vals = gradient(x -> Tensor(value(shape, x)), x, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

function feupdate!(
        mpvalues::AbstractArray{<: MPValue{S}}, mesh::UnstructuredMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        volume::Union{Nothing, AbstractArray} = nothing,
        quadrature::Tuple = (quadpoints(cellshape(mesh)), quadweights(cellshape(mesh))),
    ) where {dim, S <: Shape{dim}}
    qpts, qwts = quadrature
    @assert length(qpts) == length(qwts)
    @assert size(mesh) == size(nodes)
    @assert size(mpvalues) == (length(qpts), ncells(mesh))
    @assert isnothing(volume) || size(mpvalues) == size(volume)
    valgrads = values.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        x = nodes[indices]
        for p in eachindex(qpts, qwts)
            N, dNdξ = valgrads[p]
            J = sum(x .⊗ dNdξ)
            set_values!(mpvalues[p,c], (N, dNdξ .⊡ Ref(inv(J))))
            neighboringnodes_storage(mpvalues[p,c])[] = indices
            if volume !== nothing
                volume[p,c] = qwts[p] * det(J)
            end
        end
    end
end

_get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
_get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
function feupdate!(
        mpvalues::AbstractArray{<: MPValue{S}}, mesh::UnstructuredMesh{S, dim},
        nodes::AbstractArray{<: Vec{dim}} = mesh;
        area::AbstractArray = nothing, normal::AbstractArray = nothing,
        quadrature::Tuple = (quadpoints(cellshape(mesh)), quadweights(cellshape(mesh))),
    ) where {S <: Shape, dim}
    qpts, qwts = quadrature
    @assert length(qpts) == length(qwts)
    @assert size(mesh) == size(nodes)
    @assert size(mpvalues) == (length(qpts), ncells(mesh))
    @assert isnothing(area) || size(mpvalues) == size(area)
    @assert isnothing(normal) || size(mpvalues) == size(normal)
    valgrads = values.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        x = nodes[indices]
        for p in eachindex(qpts, qwts)
            N, dNdξ = valgrads[p]
            J = sum(x .⊗ dNdξ)
            n = _get_normal(J)
            n_norm = norm(n)
            set_values!(mpvalues[p,c], (N,))
            neighboringnodes_storage(mpvalues[p,c])[] = indices
            if area !== nothing
                area[p,c] = qwts[p] * n_norm
            end
            if normal !== nothing
                normal[p,c] = n / n_norm
            end
        end
    end
end
