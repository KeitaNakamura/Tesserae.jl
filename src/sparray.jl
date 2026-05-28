struct SpIndex{I}
    index::I
    spindex::Int
end

# `SpIndex` is tied to the current numbering of a `SpIndices` object.
# It should be treated as a short-lived token and not stored across
# `update_sparsity!` calls.
@inline logicalindex(x::SpIndex) = x.index
@inline storageindex(x::SpIndex) = x.spindex
isactive(x::SpIndex) = !iszero(x.spindex)

@inline function Base.getindex(A::AbstractArray, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    @inbounds isactive(i) ? A[logicalindex(i)] : zero_recursive(eltype(A))
end

Base.show(io::IO, x::SpIndex) = print(io, "SpIndex(", x.index, ", ", ifelse(isactive(x), x.spindex, CDot()), ")")

# Block sparsity is stored as a dense array over block coordinates.
# Zero means inactive; positive values are compact blocknumbers for SpArray.data.
struct SpIndices{dim, L, B <: AbstractArray{Int, dim}} <: AbstractArray{SpIndex{CartesianIndex{dim}}, dim}
    dims::Dims{dim}
    blocknumbering::B
end

function SpIndices(dims::Dims{dim}; block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {dim, L}
    _check_block_size_log2(block_size_log2)
    blocknumbering = fill(0, nblocks(dims; block_size_log2))
    SpIndices{dim, L, typeof(blocknumbering)}(dims, blocknumbering)
end
SpIndices(dims::Int...; kwargs...) = SpIndices(dims; kwargs...)
SpIndices(mesh::CartesianMesh) = SpIndices(size(mesh); block_size_log2=Val(block_size_log2(mesh)))

Base.size(sp::SpIndices) = sp.dims
Base.IndexStyle(::Type{<: SpIndices}) = IndexCartesian()

@inline blocknumbering(sp::SpIndices) = sp.blocknumbering
@inline nblocks(sp::SpIndices) = size(blocknumbering(sp))
@inline block_size_log2(::SpIndices{dim, L}) where {dim, L} = L

# Each active block stores a dense block of size blocksize(sp) in SpArray.data.
@inline blockwidth(sp::SpIndices) = blockwidth(Val(block_size_log2(sp)))
@inline blocksize(sp::SpIndices{dim}) where {dim} = nfill(blockwidth(sp), Val(dim))
@inline blocklength(sp::SpIndices{dim, L}) where {dim, L} = 1 << (L*dim)

# blocknumber + local linear index inside the block -> SpArray.data index.
@inline storageindex(blocknumber::Integer, localindex::Integer, sp::SpIndices) =
    (blocknumber - 1) * blocklength(sp) + localindex

# Logical node index -> block coordinate.
@inline blockindex(I::Vararg{Integer, dim}; block_size_log2::Val{L}) where {dim, L} =
    @. ((I - 1) >> L) + 1

# Logical node index -> block coordinate and local linear index inside the block.
@inline function global_to_blocklocal(I::Vararg{Integer, dim}; block_size_log2::Val{L}) where {dim, L}
    j = I .- 1
    block = blockindex(I...; block_size_log2)
    localcoord = @. (j & ((1 << L) - 1)) + 1
    LI = LinearIndices(nfill(1 << L, Val(dim)))
    @inbounds block, LI[localcoord...]
end

# block coordinate and local Cartesian index inside the block -> logical node index.
@inline function blocklocal_to_global(block::CartesianIndex{dim}, localcoord::CartesianIndex{dim}; block_size_log2::Val{L}) where {dim, L}
    CartesianIndex(ntuple(d -> ((block[d] - 1) << L) + localcoord[d], Val(dim)))
end

@inline function Base.getindex(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    block_size = Val(block_size_log2(sp))
    block, localindex = global_to_blocklocal(I...; block_size_log2=block_size)
    @inbounds blocknumber = blocknumbering(sp)[block...]
    index = storageindex(blocknumber, localindex, sp)
    SpIndex(CartesianIndex(I), ifelse(iszero(blocknumber), zero(index), index))
end

struct ActiveSpIndices{dim, S <: SpIndices{dim}}
    spinds::S
end

# Iterate active logical indices in storage order. This is intentionally not
# Cartesian iteration order; callers that work with `SpArray.data` can use the
# resulting `SpIndex` values without re-sorting.
activeindices(sp::SpIndices) = ActiveSpIndices(sp)

Base.IteratorSize(::Type{<: ActiveSpIndices}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<: ActiveSpIndices}) = Base.HasEltype()
Base.eltype(::Type{ActiveSpIndices{dim, S}}) where {dim, S} = SpIndex{CartesianIndex{dim}}

function Base.iterate(iter::ActiveSpIndices{dim}, state=(1, 1)) where {dim}
    sp = iter.spinds
    numbering = blocknumbering(sp)
    blocks = CartesianIndices(numbering)
    localindices = CartesianIndices(blocksize(sp))
    block_size = Val(block_size_log2(sp))
    nblock = length(numbering)
    nlocal = length(localindices)
    b, l = state

    @inbounds while b ≤ nblock
        blocknumber = numbering[b]
        if !iszero(blocknumber)
            block = blocks[b]
            while l ≤ nlocal
                localcoord = localindices[l]
                I = blocklocal_to_global(block, localcoord; block_size_log2=block_size)
                i = storageindex(blocknumber, l, sp)
                l += 1
                checkbounds(Bool, sp, Tuple(I)...) && return SpIndex(I, i), (b, l)
            end
        end
        b += 1
        l = 1
    end

    nothing
end

@inline function isactive(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    block = blockindex(I...; block_size_log2=Val(block_size_log2(sp)))
    @inbounds !iszero(blocknumbering(sp)[block...])
end
@inline isactive(sp::SpIndices, I::CartesianIndex) = (@_propagate_inbounds_meta; isactive(sp, Tuple(I)...))

# this should be done after updating `blocknumbering(sp)`
function numbering!(sp::SpIndices{dim}) where {dim}
    numbers = blocknumbering(sp)
    count = 0
    @inbounds for i in eachindex(numbers)
        numbers[i] = iszero(numbers[i]) ? 0 : (count += 1)
    end
    count * blocklength(sp) # return the number of activated nodes
end

function countnnz(sp::SpIndices)
    count(!iszero, blocknumbering(sp)) * blocklength(sp)
end

function update_sparsity!(sp::SpIndices, blkspy::AbstractArray{Bool})
    nblocks(sp) == size(blkspy) || throw(ArgumentError("blocks per dimension $(nblocks(sp)) must match"))
    blocknumbering(sp) .= blkspy
    numbering!(sp)
end

function update_sparsity!(spinds::SpIndices, partition::ColorPartition{<: BlockStrategy})
    bs = strategy(partition)
    nblocks(spinds) == nblocks(bs) || throw(ArgumentError("blocks per dimension $(nblocks(spinds)) must match"))
    block_size_log2(spinds) == block_size_log2(bs) ||
        throw(ArgumentError("block_size_log2 $(block_size_log2(spinds)) must match partition block_size_log2 $(block_size_log2(bs))"))

    inds = fillzero!(blocknumbering(spinds))
    CI = CartesianIndices(nblocks(bs))
    @inbounds for I in CI
        if !isempty(particle_indices_in(bs, I))
            blks = (I - oneunit(I)):(I + oneunit(I))
            inds[blks ∩ CI] .= true
        end
    end

    numbering!(spinds)
end

"""
    SpArray{T}(undef, dims...)

`SpArray` is a sparse array which has blockwise sparsity pattern.
In `SpArray`, it is not allowed to freely change the value like built-in `Array`.
For example, trying to `setindex!` doesn't change anything without any errors as

```jldoctest sparray
julia> A = SpArray{Float64}(undef, 5, 5)
5×5 SpArray{Float64, 2}:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> A[1,1]
0.0

julia> A[1,1] = 2 # no error
2

julia> A[1,1] # still zero
0.0
```

This is because the block where index `(1,1)` is located is not activated yet.
To activate the block, update sparsity pattern by `update_sparsity!(A, spy)`
where `spy` must have `Tesserae.nblocks(A)`.

```jldoctest sparray
julia> spy = trues(Tesserae.nblocks(A))
2×2 BitMatrix:
 1  1
 1  1

julia> update_sparsity!(A, spy) # returned value indicates the number of allocated elements in `A`.
64

julia> A .= 0;

julia> A[1,1] = 2
2

julia> A
5×5 SpArray{Float64, 2}:
 2.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
struct SpArray{T, dim} <: AbstractArray{T, dim}
    data::Vector{T}
    spinds::SpIndices{dim}
    shared_spinds::Bool
end

function SpArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}; block_size_log2::Val{L}=Val(BLOCK_SIZE_LOG2)) where {T, L}
    data = Vector{T}(undef, 0)
    spinds = SpIndices(dims; block_size_log2)
    SpArray(data, spinds, false)
end
SpArray{T}(::UndefInitializer, dims::Int...; kwargs...) where {T} = SpArray{T}(undef, dims; kwargs...)

function SpArray{T}(spinds::SpIndices) where {T}
    data = Vector{T}(undef, 0)
    SpArray(data, spinds, true)
end

Base.IndexStyle(::Type{<: SpArray}) = IndexCartesian()
Base.size(A::SpArray) = size(A.spinds)

get_data(A::SpArray) = A.data
get_spinds(A::SpArray) = A.spinds
nblocks(A::SpArray) = nblocks(get_spinds(A))
storedindices(A::SpArray) = eachindex(get_data(A))
activeindices(A::SpArray) = activeindices(get_spinds(A))

# return zero if the index is not active
@inline function Base.getindex(A::SpArray, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    isactive(i) || return zero_recursive(eltype(A))
    @debug checkbounds(get_data(A), storageindex(i))
    @inbounds get_data(A)[storageindex(i)]
end

# do nothing if the index is not active (do not throw error!!)
@inline function Base.setindex!(A::SpArray, v, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    isactive(i) || return A
    @debug checkbounds(get_data(A), storageindex(i))
    @inbounds get_data(A)[storageindex(i)] = v
    A
end

@inline function Base.getindex(A::SpArray{<: Any, dim}, I::Vararg{Integer, dim}) where {dim}
    @_propagate_inbounds_meta
    A[get_spinds(A)[I...]]
end

@inline function Base.setindex!(A::SpArray{<: Any, dim}, v, I::Vararg{Integer, dim}) where {dim}
    @_propagate_inbounds_meta
    A[get_spinds(A)[I...]] = v
    A
end

@inline function add!(A::SpArray{T}, i::SpIndex, v::T) where {T}
    @boundscheck checkbounds(A, logicalindex(i))
    isactive(i) || return A
    @debug checkbounds(get_data(A), storageindex(i))
    @inbounds get_data(A)[storageindex(i)] += v
    A
end

@inline isactive(A::SpArray, I...) = (@_propagate_inbounds_meta; isactive(get_spinds(A), I...))

fillzero!(A::SpArray) = (fillzero!(A.data); A)

function update_sparsity!(A::SpArray, blkspy)
    A.shared_spinds && error("""
    The sparsity pattern is shared among some `SpArray`s. \
    Perhaps you should use `update_sparsity!(grid, blkspy)` instead of applying it to each `SpArray`.
    """)
    n = update_sparsity!(get_spinds(A), blkspy)
    resize_fillzero_data!(A, n)
    n
end

function resize_fillzero_data!(A::SpArray, n::Integer)
    fillzero!(resize!(get_data(A), n))
    A
end
resize_fillzero_data!(A::AbstractMesh, n) = A

#############
# Broadcast #
#############

Broadcast.BroadcastStyle(::Type{<: SpArray}) = ArrayStyle{SpArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{SpArray}}, ::Type{ElType}) where {ElType}
    N = ndims(bc)
    bc′ = convert(Broadcasted{DefaultArrayStyle{N}}, bc)
    similar(bc′, ElType)
end

function Base.copyto!(dest::SpArray, bc::Broadcasted{ArrayStyle{SpArray}})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bcf = Broadcast.flatten(bc)
    if identical_spinds(dest, bcf.args...)
        Base.copyto!(_get_data(dest), _get_data(bc))
    else
        @inbounds for i in activeindices(dest)
            dest[i] = bc[logicalindex(i)]
        end
    end
    dest
end
@inline _get_data(bc::Broadcasted{ArrayStyle{SpArray}}) = Broadcast.broadcasted(bc.f, map(_get_data, bc.args)...)
@inline _get_data(x::SpArray) = get_data(x)
@inline _get_data(x::Any) = x

# helpers for copyto!
# all abstract arrays except SpArray and Tensor are not allowed in broadcasting
_ok(::Type{<: AbstractArray}) = false
_ok(::Type{<: SpArray}) = true
_ok(::Type{<: Tensor})  = true
_ok(::Type{<: Any})     = true
@generated function identical_spinds(args...)
    all(_ok, args) || return :(false)
    exps = [:(args[$i].spinds) for i in 1:length(args) if args[i] <: SpArray]
    n = length(exps)
    quote
        spindss = tuple($(exps...))
        @nall $n i -> spindss[1] === spindss[i]
    end
end

###############
# Custom show #
###############

struct CDot end
Base.show(io::IO, x::CDot) = print(io, "⋅")

struct ShowSpArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::A
end
Base.size(x::ShowSpArray) = size(x.parent)
Base.axes(x::ShowSpArray) = axes(x.parent)
@inline function Base.getindex(x::ShowSpArray, i::Integer...)
    @_propagate_inbounds_meta
    p = x.parent
    isactive(get_spinds(p)[i...]) ? maybecustomshow(p[i...]) : CDot()
end
maybecustomshow(x) = x
maybecustomshow(x::SpArray) = ShowSpArray(x)

Base.summary(io::IO, x::ShowSpArray) = summary(io, x.parent)
Base.show(io::IO, mime::MIME"text/plain", x::SpArray) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpArray) = show(io, ShowSpArray(x))
