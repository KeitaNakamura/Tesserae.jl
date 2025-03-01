@inline function Base.values(::Order{1}, shape::Shape, x::Vec)
    grads, vals = gradient(x -> Tensor(value(shape, x)), x, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

function feupdate!(
        mpvalues::AbstractArray{<: MPValue{S}}, detJdVₚ::AbstractArray, mesh::UnstructuredMesh{S};
        node_coordinates::AbstractArray{<: Vec} = mesh, quadrature::Tuple = (quadpoints(cellshape(mesh)), quadweights(cellshape(mesh))),
    ) where {S <: Shape}
    qpts, qwts = quadrature
    @assert length(qpts) == length(qwts)
    @assert size(mesh) == size(node_coordinates)
    @assert size(mpvalues) == size(detJdVₚ) == (length(qpts), ncells(mesh))
    valgrads = values.(Ref(Order(1)), Ref(cellshape(mesh)), qpts)
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        x = node_coordinates[indices]
        for p in eachindex(qpts, qwts)
            N, dNdξ = valgrads[p]
            J = sum(x .⊗ dNdξ)
            set_values!(mpvalues[p,c], (N, dNdξ .⊡ Ref(inv(J))))
            detJdVₚ[p,c] = qwts[p] * det(J)
            neighboringnodes_storage(mpvalues[p,c])[] = indices
        end
    end
end
