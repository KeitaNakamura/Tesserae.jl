using KernelAbstractions: @index
using Base.Checked: checked_mul
using LinearAlgebra: Transpose, Adjoint
using SparseArrays: issparse

"""
    create_sparse_matrix(rows::Region, columns::Region, offsets; ndofs)
    create_sparse_matrix(T, rows::Region, columns::Region, offsets; ndofs)

Allocate a zero-valued fixed sparse pattern. Each offset specifies a column
position relative to a row position. Both regions must span the whole domain;
pairs outside it are omitted. `ndofs` is a positive integer or a pair of row
and column DoF counts, with components contiguous at each sample point.

For example, `(0ex, -ex, ex, -ey, ey)` describes a five-point cell stencil.
Use `@stencil_assemble` to write its coefficients and `fillzero!` to reuse it.
"""
function create_sparse_matrix(rows::Region, columns::Region, offsets; ndofs)
    create_sparse_matrix(Float64, rows, columns, offsets; ndofs)
end

function create_sparse_matrix(::Type{T}, rows::Region{N}, columns::Region{N}, offsets; ndofs) where {T, N}
    checkframe(rows.frame, columns.frame)
    rows.cells == columns.cells || throw(DimensionMismatch("matrix regions must share the same domain"))
    iswhole(rows) && iswhole(columns) || throw(ArgumentError("sparse patterns require whole-domain regions"))
    dofs = ndofs isa Int ? (ndofs, ndofs) : ndofs
    dofs isa Tuple{Int, Int} && all(>(0), dofs) || throw(ArgumentError("DoF counts must be a positive Int or a pair of positive Ints"))
    nr, nc = dofs
    nrows = checked_mul(nr, reduce(checked_mul, size(rows)))
    ncolumns = checked_mul(nc, reduce(checked_mul, size(columns)))
    row_ids, column_ids = LinearIndices(size(rows)), LinearIndices(size(columns))
    I, J = Int[], Int[]
    for offset in offsets
        offset isa GridOffset{N} || throw(ArgumentError("offset dimension must match the matrix regions"))
        (rows.location+offset) == columns.location || throw(ArgumentError("offset does not reach the column placement"))
        selected = intersect(rows, columns-offset)
        shifted = selected+offset
        for (i, j) in zip(CartesianIndices(selected.indices), CartesianIndices(shifted.indices))
            append_dofs!(I, J, node_dofs(row_ids[i], nr), node_dofs(column_ids[j], nc), 0, 0)
        end
    end
    sparse(I, J, zeros(T, length(I)), nrows, ncolumns)
end

function matrix_dofs(A, rows, columns)
    checkframe(rows.frame, columns.frame)
    rows.cells == columns.cells || throw(DimensionMismatch("matrix regions must share the same domain"))
    row_size = samplecounts(rows.location, rows.cells)
    column_size = samplecounts(columns.location, columns.cells)
    nr, rr = divrem(size(A, 1), prod(row_size))
    nc, rc = divrem(size(A, 2), prod(column_size))
    iszero(rr) && iszero(rc) && nr > 0 && nc > 0 || throw(DimensionMismatch("matrix dimensions do not match the region DoFs"))
    nr, nc, LinearIndices(row_size), LinearIndices(column_size)
end

function matrix_values(A::AbstractMatrix)
    storage = matrix_parent(A)
    storage === A || return matrix_values(storage)
    issparse(A) && throw(ArgumentError("unsupported sparse matrix storage: $(typeof(A))"))
    vec(A)
end
matrix_values(A::SparseMatrixCSC) = nonzeros(A)
matrix_values(A::Transpose{<:Any,<:AbstractMatrix}) = vec(transpose(matrix_values(parent(A))))
matrix_values(A::Adjoint{<:Any,<:AbstractMatrix}) = vec(adjoint(matrix_values(parent(A))))
function matrix_slot(A::AbstractMatrix, i, j)
    storage = matrix_parent(A)
    storage === A && return LinearIndices(A)[i, j]
    rows, columns = matrix_parent_indices(A)
    matrix_slot(storage, mapped_index(rows, i), mapped_index(columns, j))
end

function matrix_slot(A::SparseMatrixCSC, i, j)
    entries = nzrange(A, j)
    k = searchsortedfirst(rowvals(A), i, first(entries), last(entries), Base.Order.Forward)
    k ≤ last(entries) && rowvals(A)[k] == i || throw(ArgumentError("matrix entry ($i, $j) is absent from the fixed sparse pattern"))
    k
end

matrix_slot(A::Union{Transpose{T,M}, Adjoint{T,M}}, i, j) where {T,M<:AbstractMatrix} = matrix_slot(parent(A), j, i)

function matrix_slots!(buffer, A, row_nodes, column_nodes, nr, nc)
    dims = (nr*nc, size(row_nodes)...)
    slots = if isnothing(buffer)
        Array{Int}(undef, dims)
    else
        count = checked_mul(checked_mul(nr, nc), length(row_nodes))
        length(buffer) < count && resize!(buffer, count)
        reshape(view(buffer, 1:count), dims)
    end
    k = 0
    @inbounds for K in CartesianIndices(row_nodes), b in 1:nc, a in 1:nr
        i = nr*(row_nodes[K]-1)+a
        j = nc*(column_nodes[K]-1)+b
        k += 1
        slots[k] = matrix_slot(A, i, j)
    end
    slots
end

# Validate the effective paired selection before writing. Region bounds and
# right-hand-side reads use the same common-range rules as @stencil.
function stencil_assemble!(op, A::AbstractMatrix, rows::Region{N}, columns::Region{N}, rhs, buffer=nothing) where {N}
    size(rows) == size(columns) || throw(DimensionMismatch("matrix row and column selections must have equal shapes"))
    nr, nc, row_ids, column_ids = matrix_dofs(A, rows, columns)
    row_operand, column_operand = RegionOperand(row_ids, rows), RegionOperand(column_ids, columns)
    bc = rhs isa Broadcast.Broadcasted ? rhs : Broadcast.broadcasted(identity, rhs)
    checkframes(rows.frame, bc)
    checked = Broadcast.instantiate(Broadcast.Broadcasted(bc.f, bc.args, axes(row_operand)))
    bounds = restrict(restrict(available(row_operand), column_operand), checked)
    any(isempty, bounds) && return A
    row_nodes, column_nodes = lower(row_operand, bounds), lower(column_operand, bounds)
    values = matrix_values(A)
    backend = KernelAbstractions.get_backend(values)
    if backend isa KernelAbstractions.CPU
        slots = matrix_slots!(buffer, A, row_nodes, column_nodes, nr, nc)
    else
        host_slots = matrix_slots!(nothing, A, row_nodes, column_nodes, nr, nc)
        slots = similar(values, Int, size(host_slots))
        copyto!(slots, host_slots)
    end
    args = map(x -> lower(x, bounds), checked.args)
    expression = Broadcast.Broadcasted(checked.f, args, axes(row_nodes))
    assemble_entries!(backend, values, slots, expression, op, nr, nc)
    A
end

# Specialize numerical execution after resolving the native array and view types.
@noinline function assemble_entries!(backend, values, slots, expression, op, nr, nc)
    # Slots can reorder writes even when the input is the values array itself.
    destination = @inbounds view(values, slots)
    expression = Broadcast.preprocess(destination, expression)
    assemble_entries_kernel!(backend)(values, slots, expression, op, nr, nc; ndrange=map(length, axes(expression)))
end

KernelAbstractions.@kernel function assemble_entries_kernel!(values, slots, expression, op, nr, nc)
    I = @index(Global, Cartesian)
    value = expression[I]
    @inline check_matrix_entry_size(value, nr, nc)
    @inbounds for c in 1:length(value)
        k = slots[c, Tuple(I)...]
        component = value[c]
        values[k] = op === identity ? component : op(values[k], component)
    end
end

# Capture a fused RHS before Julia materializes it, including when all its
# operands are ordinary arrays. Assembly selects the valid points first.
struct AssemblyRHS{T}
    value::T
end
Broadcast.broadcasted(::Type{AssemblyRHS}, value) = AssemblyRHS(value)

function assembly_equations(ex, source, buffer=nothing)
    ex isa LineNumberNode && return ex
    ex isa Expr || throw(ArgumentError("@stencil_assemble expects a matrix update or a block of updates"))
    ex.head === :block && return Expr(:block, map(x -> assembly_equations(x, source, buffer), ex.args)...)
    ex.head in (:(=), :(+=), :(-=)) || throw(ArgumentError("matrix assembly supports =, += and -="))
    lhs, rhs = ex.args
    lhs isa Expr && lhs.head === :ref && length(lhs.args) == 3 ||
        throw(ArgumentError("matrix assembly requires A[rows, columns] on the left-hand side"))
    A, rows, columns = lhs.args
    op = ex.head === :(=) ? identity : ex.head === :(+=) ? (+) : (-)
    dotted = Expr(:macrocall, GlobalRef(Base.Broadcast, Symbol("@__dot__")), source, common_refs(rhs))
    values = Expr(:macrocall, GlobalRef(Base, Symbol("@views")), source, :($AssemblyRHS.($dotted).value))
    Expr(:call, stencil_assemble!, op, A, rows, columns, values, buffer)
end

"""
    @stencil_assemble A[rows, columns] += coefficient
    @stencil_assemble begin
        equations...
    end

Pair corresponding points of two equally shaped Regions and update their
matrix entries. This is pointwise pairing, not a Cartesian product. Numerical
expressions use `@views @.` and common-range selection, as in `@stencil`.
Equations execute in order; `=`, `+=` and `-=` replace, add and subtract values.
Missing sparse entries are errors; the matrix structure is never extended.
For multiple DoFs, each pointwise value is a local matrix (or a vector when
one side has one DoF), evaluated once per selected point before its components
are written. Clear the matrix explicitly before a fresh assembly.
CPU equations in a block reuse one temporary position buffer. Every equation
still checks and searches its own entries before updating any matrix values.

```julia
A = create_sparse_matrix(C, C, (0ex, -ex, ex); ndofs=1)
L, R = I-ex/2, I+ex/2
@stencil_assemble begin
    A[L, L] += coefficient[I]
    A[L, R] -= coefficient[I]
    A[R, L] -= coefficient[I]
    A[R, R] += coefficient[I]
end
```
"""
macro stencil_assemble(ex)
    ex isa Expr && ex.head === :block || return esc(assembly_equations(ex, __source__))
    buffer = gensym(:slots)
    equations = assembly_equations(ex, __source__, buffer)
    esc(:(let $buffer = $(Vector{Int})()
        $equations
    end))
end
