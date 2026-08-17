# -----------------------------------------------------------------------------
#  Zeroing
# -----------------------------------------------------------------------------

@generated function zero_recursive(::Type{T}) where {T}
    isbitstype(T) || return :(throw(ArgumentError("`zero_recursive` supports only `isbitstype`, got $T")))
    :(@_inline_meta; zero(T))
end
@generated function zero_recursive(::Type{T}) where {T <: Union{Tuple, NamedTuple}}
    exps = [:(zero_recursive($t)) for t in fieldtypes(T)]
    :(@_inline_meta; T(($(exps...),)))
end
zero_recursive(x) = zero_recursive(typeof(x))

function fillzero!(x::AbstractArray)
    fill!(x, zero_recursive(eltype(x)))
    x
end

# `fill!` stores a composite element one field at a time, slower than a memset
# over the same bytes, and grid fields are mostly composite.
#
# The test is structural rather than an inspection of a zero value, since reading
# the bytes of one would also read padding, which is undefined. A type built only
# from numbers but with a nonzero `zero` would take this path wrongly, but
# `zero_recursive` reaches leaves the same way, so it is already outside what
# `fillzero!` promises. Padding is fine: memset zeroes it and nothing reads it.
Base.@assume_effects :foldable function zeroed_by_memset(::Type{T}) where {T}
    isbitstype(T) && sizeof(T) > 0 || return false
    T <: Union{Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
               Int128, UInt128, Float16, Float32, Float64} && return true
    isprimitivetype(T) && return false # unknown primitive: no zero to assume
    fieldcount(T) > 0 && all(zeroed_by_memset, fieldtypes(T))
end

function fillzero!(x::Array{T}) where {T}
    if zeroed_by_memset(T)
        GC.@preserve x Libc.memset(Ptr{UInt8}(pointer(x)), 0, sizeof(T) * length(x))
    else
        fill!(x, zero_recursive(T))
    end
    x
end

function fillzero!(x::StructArray)
    StructArrays.foreachfield(fillzero!, x)
    x
end

function resize_fillzero!(A::AbstractVector, n::Integer)
    # Existing Metal buffers cannot resize to zero length. Zeroing stale
    # storage is enough when the active numbering no longer references it.
    if iszero(n) && get_device(A) isa MetalDevice
        fillzero!(A)
    else
        fillzero!(resize!(A, n))
    end
    A
end

# A byte range is the only thing that can be split across threads, so a target
# without one reports `nothing` and the caller zeroes it itself.
memset_buffer(x) = nothing
memset_buffer(x::Array{T}) where {T} = zeroed_by_memset(T) ? x : nothing

# Dispatch rather than a test over the mapped tuple, so falling back is a matter
# of which method is called and not of the compiler folding an `any`.
memset_buffers(targets::Tuple) = _memset_buffers(map(memset_buffer, targets))
_memset_buffers(buffers::Tuple{Vararg{Array}}) = buffers
_memset_buffers(::Tuple) = nothing

# Chunk boundaries are aligned so workers do not share the line they write. A
# machine with wider lines still gets correct results, just sharing at the seams.
const FILLZERO_CHUNK_ALIGN = 64

# Unrolled rather than a `foreach`, so zeroing on one thread stays a run of
# inline `fillzero!` calls.
@inline fillzero_each!(::Tuple{}) = nothing
@inline fillzero_each!(targets::Tuple) = (fillzero!(first(targets)); fillzero_each!(Base.tail(targets)))

# The GPU counterpart of `fillzero_prologue`: one launch zeroes every target,
# where a `fillzero!` per field costs one launch each.
zero_buffer(A::AbstractArray) = A

@inline _fillzero_at!(::Tuple{}, I) = nothing
@inline function _fillzero_at!(buffers::Tuple, I)
    buffer = first(buffers)
    if I <= length(buffer)
        @inbounds buffer[I] = zero_recursive(eltype(buffer))
    end
    _fillzero_at!(Base.tail(buffers), I)
end

@kernel function gpukernel_fillzero_each(buffers)
    I = @index(Global)
    _fillzero_at!(buffers, I)
end

fillzero_each!(::GPUDevice, ::Tuple{}) = nothing
function fillzero_each!(device::GPUDevice, targets::Tuple)
    buffers = map(zero_buffer, targets)
    kernel = gpukernel_fillzero_each(get_backend(device))
    kernel(buffers; ndrange=maximum(length, buffers))
    nothing
end

"""
    fillzero_prologue(targets::Tuple)

A `prologue` for `partitioned_foreach` that zeroes `targets`, or `nothing` when
they cannot be zeroed that way and the caller has to do it itself. The targets
are split as one concatenated byte range, so fields of unequal size still give
the workers even shares.
"""
fillzero_prologue(::Tuple{}) = nothing
fillzero_prologue(targets::Tuple) = _fillzero_prologue(memset_buffers(targets))

_fillzero_prologue(::Nothing) = nothing
function _fillzero_prologue(buffers::Tuple)
    nbytes = sum(sizeof, buffers)
    (nchunks, chunk_id) -> fillzero_chunk!(buffers, nbytes, nchunks, chunk_id)
end

function fillzero_chunk!(buffers::Tuple, nbytes::Int, nchunks::Int, chunk_id::Int)
    chunksize = FILLZERO_CHUNK_ALIGN * cld(nbytes, FILLZERO_CHUNK_ALIGN * nchunks)
    fillzero_byte_range!(buffers, chunk_range(chunk_id, chunksize, nbytes), 0)
end

# `range` indexes the buffers laid end to end, `offset` counting the bytes the
# buffers ahead of this one take up.
@inline fillzero_byte_range!(::Tuple{}, range::UnitRange{Int}, offset::Int) = nothing
@inline function fillzero_byte_range!(buffers::Tuple, range::UnitRange{Int}, offset::Int)
    buffer = first(buffers)
    nbytes = sizeof(buffer)
    from = max(first(range) - offset, 1)
    to = min(last(range) - offset, nbytes)
    if from ≤ to
        GC.@preserve buffer Libc.memset(Ptr{UInt8}(pointer(buffer)) + (from-1), 0, to-from+1)
    end
    fillzero_byte_range!(Base.tail(buffers), range, offset + nbytes)
end
