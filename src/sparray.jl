struct BlockSparsity{dim, Tindices <: AbstractArray{<: Integer, dim}} <: AbstractArray{Bool, dim}
    blkinds::Tindices
end
Base.size(spy::BlockSparsity) = size(spy.blkinds)
Base.IndexStyle(::Type{<: BlockSparsity}) = IndexLinear()
@inline function Base.getindex(spy::BlockSparsity, i::Integer)
    @boundscheck checkbounds(spy, i)
    @inbounds !isnullindex(spy.blkinds[i])
end
@inline function Base.setindex!(spy::BlockSparsity, v, i::Integer)
    @boundscheck checkbounds(spy, i)
    @inbounds spy.blkinds[i] = convert(Bool, v)
    spy
end

isnullindex(i::Integer) = iszero(i)
nullindex(i::Integer) = zero(i)

struct SpIndices{dim, Tindices <: AbstractArray{<: Integer, dim}} <: AbstractArray{Int, dim}
    dims::Dims{dim}
    blkinds::Tindices
end

SpIndices(dims::Tuple{Vararg{Int}}) = SpIndices(dims, fill(UInt32(0), blocksize(dims)))
SpIndices(dims::Int...) = SpIndices(dims)

Base.size(sp::SpIndices) = sp.dims
Base.IndexStyle(::Type{<: SpIndices}) = IndexCartesian()

@inline blockindices(sp::SpIndices) = sp.blkinds

@inline function _blocklocal(I::Vararg{Integer, dim}) where {dim}
    j = I .- 1
    blk = @. j >> BLOCKFACTOR + 1
    lcl = @. j & (1<<BLOCKFACTOR - 1) + 1
    blk, lcl
end
@inline function blocklocal(I::Vararg{Integer, dim}) where {dim}
    blk, lcl = _blocklocal(I...)
    LI = LinearIndices(nfill(1 << BLOCKFACTOR, Val(dim)))
    @inbounds blk, LI[lcl...]
end

@inline function Base.getindex(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    blk, lcl = blocklocal(I...)
    @inbounds begin
        n = blockindices(sp)[blk...]
        index = (n-1) << (BLOCKFACTOR*dim) + lcl
        ifelse(isnullindex(n), nullindex(index), index)
    end
end

@inline function isnonzero(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    blkinds = blockindices(sp)
    @inbounds !isnullindex(blkinds[blocksize(I)...])
end

function numbering!(sp::SpIndices{dim}) where {dim}
    inds = blockindices(sp)
    count = 0
    @inbounds for i in eachindex(inds)
        inds[i] = isnullindex(inds[i]) ? 0 : (count += 1)
    end
    count << (BLOCKFACTOR*dim)
end

function update_sparsity!(sp::SpIndices, spy_blk::AbstractArray{Bool})
    blocksparsity(sp) .= spy_blk
    numbering!(sp)
end

blocksparsity(sp::SpIndices) = BlockSparsity(blockindices(sp))

"""
    SpArray{T}(dims...)

`SpArray` is a sparse array which has blockwise sparsity pattern.
In `SpArray`, it is not allowed to freely change the value like built-in `Array`.
For example, trying to `setindex!` doesn't change anything without any errors as

```jldoctest sparray
julia> A = Marble.SpArray{Float64}(5,5)
5×5 Marble.SpArray{Float64, 2, Vector{Float64}, Matrix{UInt32}}:
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
where `spy` must have `blocksize(A)`.

```jldoctest sparray
julia> spy = falses(blocksize(A))
1×1 BitMatrix:
 0

julia> spy[1,1] = true
true

julia> update_sparsity!(A, spy)
5×5 Marble.SpArray{Float64, 2, Vector{Float64}, Matrix{UInt32}}:
 2.23145e-314  2.61586e-314  2.61723e-314  2.61675e-314  2.94553e-314
 2.37e-322     2.61623e-314  3.02298e-314  2.61586e-314  2.94543e-314
 2.7826e-318   2.61586e-314  2.94156e-314  2.61586e-314  3.05631e-314
 2.37e-322     2.61512e-314  2.94553e-314  2.61587e-314  2.61675e-314
 2.96e-322     2.61615e-314  2.94543e-314  2.61512e-314  2.61586e-314

julia> A[1,1] = 2
2

julia> A
5×5 Marble.SpArray{Float64, 2, Vector{Float64}, Matrix{UInt32}}:
 2.0           2.61586e-314  2.61723e-314  2.61675e-314  2.94553e-314
 2.37e-322     2.61623e-314  3.02298e-314  2.61586e-314  2.94543e-314
 2.7826e-318   2.61586e-314  2.94156e-314  2.61586e-314  3.05631e-314
 2.37e-322     2.61512e-314  2.94553e-314  2.61587e-314  2.61675e-314
 2.96e-322     2.61615e-314  2.94543e-314  2.61512e-314  2.61586e-314
```
"""
struct SpArray{T, dim, V <: AbstractVector{T}, A} <: AbstractArray{T, dim}
    data::V
    spinds::SpIndices{dim, A}
    shared_spinds::Bool
end

function SpArray{T}(dims::Tuple{Vararg{Int}}) where {T}
    data = Vector{T}(undef, 0)
    spinds = SpIndices(dims)
    SpArray(data, spinds, false)
end
SpArray{T}(dims::Int...) where {T} = SpArray{T}(dims)

function SpArray{T}(spinds::SpIndices) where {T}
    data = Vector{T}(undef, 0)
    SpArray(data, spinds, true)
end

Base.IndexStyle(::Type{<: SpArray}) = IndexCartesian()
Base.size(A::SpArray) = size(A.spinds)

SparseArrays.nonzeros(A::SpArray) = A.data
get_spinds(A::SpArray) = A.spinds

# return zero if the index is not active
@inline function Base.getindex(A::SpArray{<: Any, dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    spinds = get_spinds(A)
    @inbounds begin
        index = spinds[I...]
        isnullindex(index) ? zero_recursive(eltype(A)) : nonzeros(A)[index]
    end
end

# do nothing if the index is not active (don't throw error!!)
@inline function Base.setindex!(A::SpArray{<: Any, dim}, v, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    spinds = get_spinds(A)
    @inbounds begin
        index = spinds[I...]
        isnullindex(index) && return A
        nonzeros(A)[index] = v
    end
    A
end

@inline isnonzero(A::SpArray, I::Integer...) = (@_propagate_inbounds_meta; isnonzero(get_spinds(A), I...))
@inline isnonzero(A::SpArray, I::CartesianIndex) = (@_propagate_inbounds_meta; isnonzero(A, Tuple(I)...))

fillzero!(A::SpArray) = (fillzero!(A.data); A)

function update_sparsity!(A::SpArray, spy::AbstractArray{Bool})
    A.shared_spinds && error("SpArray: `update_sparsity!` should be done in `update!` for `MPSpace`. Don't call this manually.")
    n = update_sparsity!(get_spinds(A), spy)
    resize_nonzeros!(A, n)
    n
end

function resize_nonzeros!(A::SpArray, n::Integer)
    nz = nonzeros(A)
    len = length(nz)
    if 10n < len || len < n
        resize!(nz, 2n)
    end
    A
end
resize_nonzeros!(A, n) = A

###############################
# NonzeroIndex/NonzeroIndices #
###############################

struct NonzeroIndex{I}
    index::I
    nzindex::Int
end
Base.convert(::Type{I}, nz::NonzeroIndex{I}) where {I} = nz.index
@inline function Base.getindex(A::SpArray, nz::NonzeroIndex)
    @boundscheck checkbounds(nonzeros(A), nz.nzindex)
    @inbounds nonzeros(A)[nz.nzindex]
end
@inline function Base.setindex!(A::SpArray, v, nz::NonzeroIndex)
    @boundscheck checkbounds(nonzeros(A), nz.nzindex)
    @inbounds nonzeros(A)[nz.nzindex] = v
    A
end
@inline function Base.getindex(A::AbstractArray, nz::NonzeroIndex)
    @boundscheck checkbounds(A, nz.index)
    @inbounds A[nz.index]
end
@inline function Base.setindex!(A::AbstractArray, v, nz::NonzeroIndex)
    @boundscheck checkbounds(A, nz.index)
    @inbounds A[nz.index] = v
    A
end

@inline function nonzeroindex(inds::SpIndices, index)
    @boundscheck checkbounds(inds, index)
    @inbounds NonzeroIndex(index, inds[index])
end

struct NonzeroIndices{I, dim, Tparent <: AbstractArray{I, dim}, Tspinds <: SpIndices{dim}} <: AbstractArray{NonzeroIndex{I}, dim}
    parent::Tparent
    spinds::Tspinds
end
Base.parent(x::NonzeroIndices) = x.parent
Base.size(x::NonzeroIndices) = size(parent(x))
get_spinds(x::NonzeroIndices) = x.spinds
@inline function Base.getindex(x::NonzeroIndices, I...)
    @boundscheck checkbounds(x, I...)
    @inbounds begin
        index = parent(x)[I...]
        nonzeroindex(get_spinds(x), index)
    end
end
@inline function nonzeroindices(spinds::SpIndices, inds)
    @boundscheck checkbounds(spinds, inds)
    NonzeroIndices(inds, spinds)
end

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
        Base.copyto!(_nonzeros(dest), _nonzeros(bc))
    else
        @inbounds @simd for I in eachindex(dest)
            if isnonzero(dest, I)
                dest[I] = bc[I]
            end
        end
    end
    dest
end
@inline _nonzeros(bc::Broadcasted{ArrayStyle{SpArray}}) = Broadcast.broadcasted(bc.f, map(_nonzeros, bc.args)...)
@inline _nonzeros(x::SpArray) = nonzeros(x)
@inline _nonzeros(x::Any) = x

# helpers for copyto!
# all abstract arrays except SpArray and Tensor are not allowed in broadcasting
_ok(::Type{<: AbstractArray}) = false
_ok(::Type{<: SpArray})       = true
_ok(::Type{<: Tensor})        = true
_ok(::Type{<: Any})           = true
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
    isnullindex(get_spinds(p)[i...]) ? CDot() : maybecustomshow(p[i...])
end
maybecustomshow(x) = x
maybecustomshow(x::SpArray) = ShowSpArray(x)

Base.summary(io::IO, x::ShowSpArray) = summary(io, x.parent)
Base.show(io::IO, mime::MIME"text/plain", x::SpArray) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpArray) = show(io, ShowSpArray(x))
