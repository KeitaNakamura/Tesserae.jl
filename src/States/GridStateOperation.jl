struct GridStateOperation{dim, C <: AbstractCollection}
    indices::DofMapIndices{dim}
    dofindices::Vector{Vector{Int}}
    nzval::C
end

indices(x::GridStateOperation) = x.indices
dofindices(x::GridStateOperation) = x.dofindices
nonzeros(x::GridStateOperation) = x.nzval

_collection(x::AbstractVector) = Collection(x)
_collection(x::AbstractCollection{1}) = x


##############
# operations #
##############

const UnionGridState = Union{GridState, GridStateOperation}

# checkspace
checkspace(::Type{Bool}, x::UnionGridState, y::UnionGridState) = (indices(x) === indices(y)) && (dofindices(x) === dofindices(y))
checkspace(::Type{Bool}, x::UnionGridState, y::UnionGridState, zs::UnionGridState...) =
    checkspace(Bool, x, y) ? checkspace(Bool, x, zs...) : false
function checkspace(x::UnionGridState, y::UnionGridState, zs::UnionGridState...)
    checkspace(Bool, x, y, zs...) && return nothing
    throw(ArgumentError("grid states are not in the same space"))
end
# indices/dofindices:
#   Checkspace if their spaces are identical, and return indices/dofindices.
indices(x::UnionGridState, y::UnionGridState, zs::UnionGridState...) = (checkspace(x, y, zs...); indices(x))
dofindices(x::UnionGridState, y::UnionGridState, zs::UnionGridState...) = (checkspace(x, y, zs...); dofindices(x))

for op in (:+, :-, :/, :*)
    @eval begin
        Base.$op(x::UnionGridState, y::UnionGridState) =
            GridStateOperation(indices(x, y), dofindices(x, y), $op(_collection(nonzeros(x)), _collection(nonzeros(y))))
    end
    if op == :* || op == :/
        @eval begin
            Base.$op(x::UnionGridState, y::Number) = GridStateOperation(indices(x), dofindices(x), $op(_collection(nonzeros(x)), y))
            Base.$op(x::Number, y::UnionGridState) = GridStateOperation(indices(y), dofindices(y), $op(x, _collection(nonzeros(y))))
        end
    end
end

function set!(x::GridState, y::UnionGridState)
    checkspace(x, y)
    resize!(x) # should not use zeros! for incremental calculation
    nonzeros(x) .= nonzeros(y)
    x
end
