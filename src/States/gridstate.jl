struct GridState{dim, T} <: AbstractArray{T, dim}
    nzval::Vector{T}
    indices::DofMapIndices{dim}
    dofindices::Vector{Vector{Int}}
end

function gridstate(::Type{T}, dofmap::DofMap, dofindices::Vector{Vector{Int}}) where {T}
    nzval = zeros(T, ndofs(dofmap))
    GridState(nzval, indices(dofmap), dofindices)
end

Base.size(A::GridState) = size(indices(A))
nonzeros(S::GridState) = S.nzval
indices(A::GridState) = A.indices
dofindices(S::GridState) = S.dofindices
nnz(S::GridState) = ndofs(indices(S))

@inline function Base.getindex(A::GridState{dim, T}, I::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(A, I...)
    @inbounds begin
        index = indices(A)[I...]
        index === nothing ? zero(T) : A.nzval[index]
    end
end

@inline function Base.setindex!(A::GridState{dim}, v, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    @inbounds begin
        index = indices(A)[I...]
        index === nothing ? throw(ArgumentError("GridState: setindex! at invalid index")) : A.nzval[index] = v
    end
    A
end

zeros!(v::AbstractVector{T}, n) where {T} = (resize!(v, n); fill!(v, zero(T)); v)
zeros!(v) = (fill!(v, zero(eltype(v))); v)
zeros!(S::GridState) = (zeros!(nonzeros(S), nnz(S)); S)
Base.resize!(S::GridState) = (resize!(nonzeros(S), nnz(S)); S)


struct GridStateOperation{dim, C <: AbstractCollection}
    indices::DofMapIndices{dim}
    dofindices::Vector{Vector{Int}}
    nzval::C
end

indices(x::GridStateOperation) = x.indices
dofindices(x::GridStateOperation) = x.dofindices
nonzeros(x::GridStateOperation) = x.nzval

_collection(x::Vector) = Collection(x)
_collection(x::AbstractCollection{1}) = x

const UnionGridState = Union{GridState, GridStateOperation}

# for ∑ᵢ(vᵢ * N) and ∑ᵢ(vᵢ ⊗ ∇(N))
for op in (:(Base.:*), :(Tensors.:⊗))
    @eval begin
        $op(x::UnionGridState, y::AbstractCollection{2}) = $op(GridCollection(x), y)
        $op(x::AbstractCollection{2}, y::UnionGridState) = $op(x, GridCollection(y))
    end
end

for op in (:+, :-, :/, :*)
    @eval begin
        function Base.$op(x::UnionGridState, y::UnionGridState)
            GridStateOperation(indices(x, y), dofindices(x, y), $op(_collection(nonzeros(x)), _collection(nonzeros(y))))
        end
    end
    if op == :* || op == :/
        @eval begin
            function Base.$op(x::UnionGridState, y::Number)
                GridStateOperation(indices(x), dofindices(x), $op(_collection(nonzeros(x)), y))
            end
            function Base.$op(x::Number, y::UnionGridState)
                GridStateOperation(indices(y), dofindices(y), $op(x, _collection(nonzeros(y))))
            end
        end
    end
end

# checkspace
checkspace(::Type{Bool}, x::UnionGridState, y::UnionGridState) = (indices(x) === indices(y)) && (dofindices(x) === dofindices(y))
checkspace(::Type{Bool}, x::UnionGridState, y::UnionGridState, zs::UnionGridState...) =
    checkspace(Bool, x, y) ? checkspace(Bool, x, zs...) : false

function checkspace(x::UnionGridState, y::UnionGridState, zs::UnionGridState...)
    checkspace(Bool, x, y, zs...) && return nothing
    throw(ArgumentError("grid states are not in the same space"))
end

indices(x::UnionGridState, y::UnionGridState, zs::UnionGridState...) = (checkspace(x, y, zs...); indices(x))
dofindices(x::UnionGridState, y::UnionGridState, zs::UnionGridState...) = (checkspace(x, y, zs...); dofindices(x))

function set!(x::GridState, y::UnionGridState)
    checkspace(x, y)
    resize!(x) # should not use zeros! for incremental calculation
    nonzeros(x) .= nonzeros(y)
    x
end


struct GridCollection{T} <: AbstractCollection{2}
    data::T
    dofindices::Vector{Vector{Int}}
end

GridCollection(x::UnionGridState) = GridCollection(nonzeros(x), dofindices(x))

Base.length(x::GridCollection) = length(x.dofindices) # == npoints
Base.getindex(x::GridCollection, i::Int) = (@_propagate_inbounds_meta; Collection{1}(view(x.data, x.dofindices[i])))
