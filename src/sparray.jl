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
    LI = LinearIndices(nfill(1 .<< BLOCKFACTOR, Val(dim)))
    @inbounds blk, LI[lcl...]
end

@inline function Base.getindex(sp::SpIndices{dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(sp, I...)
    blk, lcl = blocklocal(I...)
    @inbounds begin
        n = blockindices(sp)[blk...]
        index = (n-1) << (BLOCKFACTOR*dim) + lcl
        ifelse(iszero(n), zero(index), index)
    end
end

reset_sparsity_pattern!(sp::SpIndices) = fillzero!(blockindices(sp))
function update_sparsity_pattern!(sp::SpIndices{dim}) where {dim}
    inds = blockindices(sp)
    count = 0
    @inbounds for i in eachindex(inds)
        inds[i] = iszero(inds[i]) ? 0 : (count += 1)
    end
    count << (BLOCKFACTOR*dim)
end

"""
    SpArray{T}(dims...)

`SpArray` is a kind of sparse array, but it is not allowed to freely change the value like `Array`.
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

julia> A[1,1] = 2
2

julia> A[1,1]
0.0
```

This is because the index `(1,1)` is not activated yet.
To activate the index, update sparsity pattern by `update_sparsity_pattern!(A, sppat)`.

```jl sparray
julia> sppat = falses(5,5); sppat[1,1] = true; sppat
5×5 BitMatrix:
 1  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> update_sparsity_pattern!(A, sppat)
5×5 Marble.SpArray{Float64, 2}:
 2.17321e-314  ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅
  ⋅            ⋅  ⋅  ⋅  ⋅

julia> A[1,1] = 2; A[1,1]
2.0
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

nonzeros(A::SpArray) = A.data
get_spinds(A::SpArray) = A.spinds

# return zero if the index is not active
@inline function Base.getindex(A::SpArray{<: Any, dim}, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    spinds = get_spinds(A)
    @inbounds begin
        index = spinds[I...]
        iszero(index) ? zero_recursive(eltype(A)) : nonzeros(A)[index]
    end
end

# do nothing if the index is not active (don't throw error!!)
@inline function Base.setindex!(A::SpArray{<: Any, dim}, v, I::Vararg{Integer, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    spinds = get_spinds(A)
    @inbounds begin
        index = spinds[I...]
        iszero(index) && return A
        nonzeros(A)[index] = v
    end
    A
end

fillzero!(A::SpArray) = (fillzero!(A.data); A)

function reset_sparsity_pattern!(A::SpArray)
    A.shared_spinds && error("SpArray: `update_sparsity_pattern!` should be done in `update!` for `MPSpace`. Don't call this manually.")
    unsafe_reset_sparsity_pattern!(A)
end

function update_sparsity_pattern!(A::SpArray)
    A.shared_spinds && error("SpArray: `update_sparsity_pattern!` should be done in `update!` for `MPSpace`. Don't call this manually.")
    unsafe_update_sparsity_pattern!(A)
end

function unsafe_reset_sparsity_pattern!(A::SpArray)
    reset_sparsity_pattern!(get_spinds(A))
end
function unsafe_update_sparsity_pattern!(A::SpArray)
    n = update_sparsity_pattern!(get_spinds(A))
    resize!(nonzeros(A), n)
    n
end

###############################
# NonzeroIndex/NonzeroIndices #
###############################

struct NonzeroIndex{I}
    parent::I
    i::Int
end
Base.parent(nz::NonzeroIndex) = nz.parent
@inline function Base.getindex(A::SpArray, nz::NonzeroIndex)
    @boundscheck checkbounds(nonzeros(A), nz.i)
    @inbounds nonzeros(A)[nz.i]
end
@inline function Base.setindex!(A::SpArray, v, nz::NonzeroIndex)
    @boundscheck checkbounds(nonzeros(A), nz.i)
    @inbounds nonzeros(A)[nz.i] = v
    A
end
@inline function Base.getindex(A::AbstractArray, nz::NonzeroIndex)
    @boundscheck checkbounds(A, parent(nz))
    @inbounds A[parent(nz)]
end
@inline function Base.setindex!(A::AbstractArray, v, nz::NonzeroIndex)
    @boundscheck checkbounds(A, parent(nz))
    @inbounds A[parent(nz)] = v
    A
end

@inline function nonzeroindex(inds::SpIndices, I)
    @boundscheck checkbounds(inds, I)
    @inbounds NonzeroIndex(I, inds[I])
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
    !identical_spinds(dest, bcf.args...) &&
        error("SpArray: broadcast along with different `SpIndices`s is not supported")
    Base.copyto!(_nonzeros(dest), _nonzeros(bc))
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
    iszero(get_spinds(p)[i...]) ? CDot() : maybecustomshow(p[i...])
end
maybecustomshow(x) = x
maybecustomshow(x::SpArray) = ShowSpArray(x)

Base.summary(io::IO, x::ShowSpArray) = summary(io, x.parent)
Base.show(io::IO, mime::MIME"text/plain", x::SpArray) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpArray) = show(io, ShowSpArray(x))
