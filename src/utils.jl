struct Index{dim}
    i::Int
    I::CartesianIndex{dim}
end
@inline Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::Index) = checkindex(Bool, inds, i.i)
@inline _to_indices(::IndexLinear, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (I[1].i, Base.tail(I)...))
@inline _to_indices(::IndexCartesian, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (Tuple(I[1].I)..., Base.tail(I)...))
@inline Base.to_indices(A, inds, I::Tuple{Index, Vararg{Any}}) = _to_indices(IndexStyle(A), A, inds, I)

nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

function Tensor3D(x::SecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[1,2], x[2,2], z, z, z, z)
end

function Tensor3D(x::SymmetricSecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SymmetricSecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[2,2], z, z)
end

function Tensor2D(x::SecondOrderTensor{3,T}) where {T}
    @inbounds SecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,1], x[2,2])
end

function Tensor2D(x::SymmetricSecondOrderTensor{3,T}) where {T}
    @inbounds SymmetricSecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,2])
end

function Tensor2D(x::FourthOrderTensor{3,T}) where {T}
    @inbounds FourthOrderTensor{2,T}((i,j,k,l) -> @inbounds(x[i,j,k,l]))
end

function Tensor2D(x::SymmetricFourthOrderTensor{3,T}) where {T}
    @inbounds SymmetricFourthOrderTensor{2,T}((i,j,k,l) -> @inbounds(x[i,j,k,l]))
end
