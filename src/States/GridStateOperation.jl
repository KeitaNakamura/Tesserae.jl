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

Base.zero(x::UnionGridState) = zero(eltype(nonzeros(x)))

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

########
# lazy #
########
lazy(op, x::UnionGridState) = GridStateOperation(indices(x), dofindices(x), lazy(op, _collection(nonzeros(x))))
function lazy(op, x::UnionGridState, y::UnionGridState)
    GridStateOperation(indices(x, y), dofindices(x, y), lazy(op, _collection(nonzeros(x)), _collection(nonzeros(y))))
end
function lazy(op, x::UnionGridState, y)
    GridStateOperation(indices(x), dofindices(x), lazy(op, _collection(nonzeros(x)), y))
end
function lazy(op, x, y::UnionGridState)
    GridStateOperation(indices(y), dofindices(y), lazy(op, x, _collection(nonzeros(y))))
end

# macros for lazy definitions
macro define_unary_operation(op)
    quote
        @inline $op(c::UnionGridState) = lazy($op, c)
    end |> esc
end
macro define_binary_operation(op)
    quote
        @inline $op(x::UnionGridState, y::UnionGridState) = lazy($op, x, y)
        @inline $op(x::UnionGridState, y) = lazy($op, x, y)
        @inline $op(x, y::UnionGridState) = lazy($op, x, y)
    end |> esc
end

# methods for number
for op in (:*, :/)
    @eval begin
        Base.$op(x::UnionGridState, y::Number) = GridStateOperation(indices(x), dofindices(x), $op(_collection(nonzeros(x)), y))
        Base.$op(x::Number, y::UnionGridState) = GridStateOperation(indices(y), dofindices(y), $op(x, _collection(nonzeros(y))))
    end
end

const unary_operations = [
    :(Base.:+),
    :(Base.:-),
    :(TensorValues.:norm),
]
const binary_operations = [
    :(Base.:+),
    :(Base.:-),
    :(Base.:/),
    :(Base.:*),
    :(TensorValues.:⋅),
    :(TensorValues.:×),
]

for op in unary_operations
    @eval @define_unary_operation $op
end
for op in binary_operations
    @eval @define_binary_operation $op
end

########
# set! #
########

function set!(x::GridState, y::UnionGridState)
    checkspace(x, y)
    resize!(x) # should not use zeros! for incremental calculation
    nzval_x = nonzeros(x)
    nzval_y = nonzeros(y)
    @assert length(nzval_x) == length(nzval_y)
    @simd for i in 1:length(nzval_x)
        @inbounds nzval_x[i] = nzval_y[i]
    end
    x
end

function set!(x::GridState, y)
    resize!(x) # should not use zeros! for incremental calculation
    nonzeros(x) .= Ref(y)
    x
end

function set!(x::GridState, y::UnionGridState, dofs::Vector{Int})
    checkspace(x, y)
    resize!(x) # should not use zeros! for incremental calculation
    view(nonzeros(x), dofs) ← view(nonzeros(y), dofs)
    x
end

function set!(x::GridState, y, dofs::Vector{Int})
    resize!(x) # should not use zeros! for incremental calculation
    fill!(view(nonzeros(x), dofs), y)
    x
end
