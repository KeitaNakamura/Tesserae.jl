# -----------------------------------------------------------------------------
#  Sparse arrays
# -----------------------------------------------------------------------------

# ---- SpArray ----

"""
    SpArray{T}(undef, dims...)

`SpArray` is a sparse array which has blockwise sparsity pattern.
In `SpArray`, it is not allowed to freely change the value like built-in `Array`.
For example, trying to `setindex!` doesn't change anything without any errors as

```jldoctest sparray
julia> A = SpArray{Float64}(undef, 5, 5)
5×5 SpArray{Float64, 2, Vector{Float64}, Tesserae.SpIndices{2, 2, Matrix{Int64}, Tesserae.BlockSparsityWorkspace{Matrix{Bool}, Matrix{Bool}, Tesserae.ParticleBlockTracker{Vector{Int64}, Matrix{Int32}}, Vector{Int64}, Vector{Int32}}}}:
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
5×5 SpArray{Float64, 2, Vector{Float64}, Tesserae.SpIndices{2, 2, Matrix{Int64}, Tesserae.BlockSparsityWorkspace{Matrix{Bool}, Matrix{Bool}, Tesserae.ParticleBlockTracker{Vector{Int64}, Matrix{Int32}}, Vector{Int64}, Vector{Int32}}}}:
 2.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
struct SpArray{T, dim, D <: AbstractVector{T}, S <: SpIndices{dim}} <: AbstractArray{T, dim}
    data::D
    spinds::S
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

function Base.fill!(A::SpArray, x)
    fill!(get_data(A), x)
    A
end

@inline function Base.getindex(A::SpArray, i::SpIndex)
    @boundscheck checkbounds(A, logicalindex(i))
    isactive(i) || return zero_recursive(eltype(A))
    @debug checkbounds(get_data(A), storageindex(i))
    @inbounds get_data(A)[storageindex(i)]
end

# Writing to an inactive index must be a no-op, not an error.
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

# Only the active nodes are stored, and they are stored contiguously, so an
# `SpArray` splits across threads exactly as the dense array behind it does.
memset_buffer(A::SpArray) = memset_buffer(get_data(A))
zero_buffer(A::SpArray) = get_data(A)

function update_sparsity!(A::SpArray, blkspy)
    A.shared_spinds && error("""
    The sparsity pattern is shared among some `SpArray`s. \
    Perhaps you should use `update_sparsity!(grid, blkspy)` instead of applying it to each `SpArray`.
    """)
    n = update_sparsity!(get_spinds(A), blkspy)
    n === nothing && (fillzero!(A); return nothing)
    resize_fillzero_data!(A, n)
    n
end

function resize_fillzero_data!(A::SpArray, n::Integer)
    resize_fillzero!(get_data(A), n)
    A
end
resize_fillzero_data!(A::AbstractMesh, n) = A

# ---- broadcast ----

# A non-mutating broadcast keeps a sparse result only for zero-preserving
# operations whose flattened arguments are all `SpArray`s sharing the very same
# `SpIndices` object (`===`), not merely an equal sparsity pattern. A mutating
# broadcast never changes sparsity: it writes into the destination's storage.

Broadcast.BroadcastStyle(::Type{<: SpArray}) = ArrayStyle{SpArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{SpArray}}, ::Type{ElType}) where {ElType}
    bc = Broadcast.instantiate(bc)
    bcf = Broadcast.flatten(bc)
    A = _first_sparray(bcf)
    _preserves_sparsity(bcf) ? similar_sparray(A, ElType) : similar(get_data(A), ElType, axes(bc))
end

similar_sparray(A::SpArray, ::Type{T}) where {T} = SpArray(similar(get_data(A), T, length(get_data(A))), get_spinds(A), true)

_first_sparray(A::SpArray) = A
_first_sparray(bc::Broadcasted) = _first_sparray(bc.args)
_first_sparray(::Tuple{}) = nothing
function _first_sparray(args::Tuple)
    A = _first_sparray(first(args))
    A === nothing ? _first_sparray(Base.tail(args)) : A
end
_first_sparray(x) = nothing

_all_sparrays(args::Tuple) = all(x -> x isa SpArray, args)
_preserves_sparsity(bc::Broadcasted) = _all_sparrays(bc.args) && identical_spinds(bc.args...) && _is_zero_preserving_bc_function(bc.f)
_is_zero_preserving_bc_function(f) = f in (+, -, *)

function Base.copyto!(dest::SpArray, bc::Broadcasted{ArrayStyle{SpArray}})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc = Broadcast.instantiate(bc)
    bcf = Broadcast.flatten(bc)
    # The shared-sparsity test needs every leaf array in one tuple, which only the
    # flattened form gives. The copy stays on the unflattened `bc`, whose nesting
    # `_get_data` rebuilds over the data arrays.
    if identical_spinds(dest, bcf.args...)
        Base.copyto!(_get_data(dest), _get_data(bc))
    else
        _copyto_sp_broadcast!(get_device(dest), dest, bc)
    end
    dest
end

function _copyto_sp_broadcast!(::CPUDevice, dest::SpArray, bc::Broadcasted)
    @inbounds for i in activeindices(dest)
        dest[i] = bc[logicalindex(i)]
    end
    dest
end

@kernel function gpukernel_copyto_sp_broadcast!(dest, @Const(bc), @Const(spinds))
    k = @index(Global)
    active, i = _active_spindex(spinds, k)
    if active
        @inbounds dest[i] = bc[logicalindex(i)]
    end
end

function _copyto_sp_broadcast!(device::GPUDevice, dest::SpArray, bc::Broadcasted)
    backend = get_backend(device)
    spinds = get_spinds(dest)
    kernel = gpukernel_copyto_sp_broadcast!(backend)
    kernel(dest, bc, spinds; ndrange=_spindex_ndrange(spinds))
    dest
end

# Instantiated here so GPU kernels do not infer broadcast axes from the sparse wrapper.
@inline _get_data(bc::Broadcasted{ArrayStyle{SpArray}}) = Broadcast.instantiate(Broadcast.broadcasted(bc.f, map(_get_data, bc.args)...))
@inline _get_data(x::SpArray) = get_data(x)
@inline _get_data(x::Any) = x

# No abstract array other than `SpArray` and `Tensor` may take the fast path.
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

# ---- custom show ----

struct CDot end
Base.show(io::IO, x::CDot) = print(io, "⋅")

struct ShowSpArray{T, N, A <: AbstractArray{T, N}, S} <: AbstractArray{T, N}
    parent::A
    summary_parent::S
end

# Array display scalar-indexes through `getindex`, so a GPU-backed array is shown
# through a CPU copy, the original being kept for the printed summary.
ShowSpArray(parent) = ShowSpArray(_show_parent(parent), parent)
_show_parent(parent) = get_device(parent) isa CPUDevice ? parent : cpu(parent)

Base.size(x::ShowSpArray) = size(x.parent)
Base.axes(x::ShowSpArray) = axes(x.parent)
@inline function Base.getindex(x::ShowSpArray, i::Integer...)
    @_propagate_inbounds_meta
    p = x.parent
    isactive(get_spinds(p)[i...]) ? maybecustomshow(p[i...]) : CDot()
end
maybecustomshow(x) = x
maybecustomshow(x::SpArray) = ShowSpArray(x)

Base.summary(io::IO, x::ShowSpArray) = summary(io, x.summary_parent)
Base.show(io::IO, mime::MIME"text/plain", x::SpArray) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpArray) = show(io, ShowSpArray(x))
