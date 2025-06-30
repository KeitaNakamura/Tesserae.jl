struct SpIndex{I}
    index::I
    spindex::Int
end

isactive(x::SpIndex) = !iszero(x.spindex)

@inline function Base.getindex(A::AbstractArray, i::SpIndex)
    @boundscheck checkbounds(A, i.index)
    @inbounds isactive(i) ? A[i.index] : zero_recursive(eltype(A))
end

Base.show(io::IO, x::SpIndex) = print(io, "SpIndex(", x.index, ", ", ifelse(isactive(x), x.spindex, CDot()), ")")

struct SpIndices{dim} <: AbstractArray{SpIndex{CartesianIndex{dim}}, dim}
    dims::Dims{dim}
    blkinds::Array{Int, dim}
end

SpIndices(dims::Tuple{Vararg{Int}}) = SpIndices(dims, fill(0, blocksize(dims)))
SpIndices(dims::Int...) = SpIndices(dims)

Base.size(sp::SpIndices) = sp.dims
Base.IndexStyle(::Type{<: SpIndices}) = IndexCartesian()

@inline blockindices(sp::SpIndices) = sp.blkinds
@inline blocksize(sp::SpIndices) = size(blockindices(sp))

@inline function _blocklocal(I::Integer...)
    j = I .- 1
    blk = @. j >> BLOCK_SIZE_LOG2 + 1
    lcl = @. j & (1<<BLOCK_SIZE_LOG2 - 1) + 1
    blk, lcl
end
@inline function blocklocal(I::Vararg{Integer, dim}) where {dim}
    blk, lcl = _blocklocal(I...)
    LI = LinearIndices(nfill(1 << BLOCK_SIZE_LOG2, Val(dim)))
    @inbounds blk, LI[lcl...]
end

@inline function Base.getindex(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    blk, lcl = blocklocal(I...)
    @inbounds n = blockindices(sp)[blk...]
    index = (n-1) << (BLOCK_SIZE_LOG2*dim) + lcl
    SpIndex(CartesianIndex(I), ifelse(iszero(n), zero(index), index))
end

@inline function isactive(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    blkinds = blockindices(sp)
    @inbounds !iszero(blkinds[blocksize(I)...])
end
@inline isactive(sp::SpIndices, I::CartesianIndex) = (@_propagate_inbounds_meta; isactive(sp, Tuple(I)...))

# this should be done after updating `blockindices(sp)`
function numbering!(sp::SpIndices{dim}) where {dim}
    inds = blockindices(sp)
    count = 0
    @inbounds for i in eachindex(inds)
        inds[i] = iszero(inds[i]) ? 0 : (count += 1)
    end
    count << (BLOCK_SIZE_LOG2*dim) # return the number of activated nodes
end

function countnnz(sp::SpIndices{dim}) where {dim}
    count(!iszero, sp.blkinds) << (BLOCK_SIZE_LOG2*dim)
end

function update_block_sparsity!(sp::SpIndices, blkspy::AbstractArray{Bool})
    blocksize(sp) == size(blkspy) || throw(ArgumentError("block size $(blocksize(sp)) must match"))
    blockindices(sp) .= blkspy
    numbering!(sp)
end

function update_block_sparsity!(spinds::SpIndices, partition::ColorPartition{<: BlockStrategy})
    bs = strategy(partition)
    blocksize(spinds) == blocksize(bs) || throw(ArgumentError("block size $(blocksize(spinds)) must match"))

    inds = fillzero!(blockindices(spinds))
    CI = CartesianIndices(blocksize(bs))
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
To activate the block, update sparsity pattern by `update_block_sparsity!(A, spy)`
where `spy` must have `Tesserae.blocksize(A)`.

```jldoctest sparray
julia> spy = trues(Tesserae.blocksize(A))
1×1 BitMatrix:
 1

julia> update_block_sparsity!(A, spy) # returned value indicates the number of allocated elements in `A`.
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

function SpArray{T}(::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T}
    data = Vector{T}(undef, 0)
    spinds = SpIndices(dims)
    SpArray(data, spinds, false)
end
SpArray{T}(::UndefInitializer, dims::Int...) where {T} = SpArray{T}(undef, dims)

function SpArray{T}(spinds::SpIndices) where {T}
    data = Vector{T}(undef, 0)
    SpArray(data, spinds, true)
end

Base.IndexStyle(::Type{<: SpArray}) = IndexCartesian()
Base.size(A::SpArray) = size(A.spinds)

get_data(A::SpArray) = A.data
get_spinds(A::SpArray) = A.spinds

# return zero if the index is not active
@inline function Base.getindex(A::SpArray, i::SpIndex)
    @boundscheck checkbounds(A, i.index)
    isactive(i) || return zero_recursive(eltype(A))
    @debug checkbounds(get_data(A), i.spindex)
    @inbounds get_data(A)[i.spindex]
end

# do nothing if the index is not active (do not throw error!!)
@inline function Base.setindex!(A::SpArray, v, i::SpIndex)
    @boundscheck checkbounds(A, i.index)
    isactive(i) || return A
    @debug checkbounds(get_data(A), i.spindex)
    @inbounds get_data(A)[i.spindex] = v
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
    @boundscheck checkbounds(A, i.index)
    isactive(i) || return A
    @debug checkbounds(get_data(A), i.spindex)
    @inbounds get_data(A)[i.spindex] += v
    A
end

@inline isactive(A::SpArray, I...) = (@_propagate_inbounds_meta; isactive(get_spinds(A), I...))

fillzero!(A::SpArray) = (fillzero!(A.data); A)

function update_block_sparsity!(A::SpArray, blkspy)
    A.shared_spinds && error("""
    The sparsity pattern is shared among some `SpArray`s. \
    Perhaps you should use `update_block_sparsity!(grid, blkspy)` instead of applying it to each `SpArray`.
    """)
    n = update_block_sparsity!(get_spinds(A), blkspy)
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
        @inbounds @simd for I in eachindex(dest)
            if isactive(dest, I)
                dest[I] = bc[I]
            end
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
