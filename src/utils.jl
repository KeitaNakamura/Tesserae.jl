# -----------------------------------------------------------------------------
#  Utilities
# -----------------------------------------------------------------------------

const DEBUG = Preferences.@load_preference("debug_mode", false)

@static if DEBUG
    @eval macro debug(ex)
        return :($(esc(ex)))
    end
else
    @eval macro debug(ex)
         return nothing
    end
end

# ---- Order/Degree ----

struct Order{n}
    Order{n}() where {n} = new{n::Int}()
end
Order(n::Int) = Order{n}()

struct Degree{n}
    Degree{n}() where {n} = new{n::Int}()
end
Degree(n::Int) = Degree{n}()
const Constant  = Degree{0}
const Linear    = Degree{1}
const Quadratic = Degree{2}
const Cubic     = Degree{3}
const Quartic   = Degree{4}
const Quintic   = Degree{5}

# ---- helpers ----

nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

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

const SparseMatrixCSCView{T, P <: SparseMatrixCSC} = SubArray{T, 2, P}

function fillzero!(matrix::SparseMatrixCSCView)
    selected_rows, selected_cols = parentindices(matrix)
    sorted_rows = issorted(selected_rows) ? selected_rows : sort(selected_rows)
    _fillzero_sparse_matrix_view!(parent(matrix), sorted_rows, selected_cols)
    matrix
end

function _fillzero_sparse_matrix_view!(matrix::SparseMatrixCSC, selected_rows, selected_cols)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    zero_value = zero_recursive(eltype(values))
    selected_stop = lastindex(selected_rows)
    @inbounds for col in selected_cols
        slots = nzrange(matrix, col)
        isempty(slots) && continue
        selected_index = searchsortedfirst(selected_rows, rows[first(slots)])
        for slot in slots
            row = rows[slot]
            while selected_index ≤ selected_stop && selected_rows[selected_index] < row
                selected_index += 1
            end
            selected_index > selected_stop && break
            if selected_rows[selected_index] == row
                values[slot] = zero_value
            end
        end
    end
    matrix
end

@inline function fastsum(f, iter)
    ret = zero(Base._return_type(f, Tuple{eltype(iter)}))
    @simd for x in iter
        ret += @inline f(x)
    end
    ret
end

commas(num::Integer) = replace(string(num), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")

getx(x) = getproperty(x, first(propertynames(x)))
getx(x::Vec) = x
getx(x::Vector{<: Vec}) = x

@generated function dropat(entries::Tuple{Vararg{Any, N}}, index::Int) where {N}
    branches = map(1:N) do i
        kept = map(j -> :(entries[$j]), filter(!=(i), 1:N))
        :(index == $i && return tuple($(kept...)))
    end
    quote
        $(branches...)
        throw(ArgumentError("index must be between 1 and tuple length"))
    end
end

struct MapArray{T, N, F, Args <: Tuple} <: AbstractArray{T, N}
    f::F
    args::Args
    function MapArray{T, N, F, Args}(f::F, args::Args) where {T, N, F, Args}
        @assert all(x->size(x)==size(first(args)), args)
        new{T, N, F, Args}(f, args)
    end
end
function maparray(f::F, args...) where {F}
    Args = map(typeof, args)
    A = Base._return_type(map, Tuple{F, Args...})
    MapArray{eltype(A), ndims(A), F, Tuple{Args...}}(f, args)
end
function maparray(f::Type{T}, args...) where {T}
    Args = map(typeof, args)
    MapArray{T, ndims(first(args)), Type{T}, Tuple{Args...}}(T, args)
end
Base.size(A::MapArray) = size(first(A.args))
Base.IndexStyle(::Type{<: MapArray}) = IndexCartesian()
@inline function Base.getindex(A::MapArray, i::Integer...)
    @boundscheck checkbounds(A, i...)
    @inbounds A.f(getindex.(A.args, i...)...)
end

struct Trues{N} <: AbstractArray{Bool, N}
    dims::Dims{N}
end
Base.size(t::Trues) = t.dims
Base.IndexStyle(::Type{<: Trues}) = IndexLinear()
@inline function Base.getindex(t::Trues, i::Integer)
    @boundscheck checkbounds(t, i)
    true
end

# ---- threaded macro ----

abstract type Scheduler end
struct StaticScheduler     <: Scheduler end
struct DynamicScheduler    <: Scheduler end
struct GreedyScheduler     <: Scheduler end
struct SequentialScheduler <: Scheduler end

get_scheduler(sch::Scheduler) = sch
get_scheduler(sch::Symbol) = get_scheduler(Val(sch))
get_scheduler(::Val{:static})  = StaticScheduler()
get_scheduler(::Val{:dynamic}) = DynamicScheduler()
get_scheduler(::Val{:greedy})  = GreedyScheduler()
get_scheduler(::Val{:nothing}) = SequentialScheduler()

# Chunks past the last piece come out empty.
@inline chunk_range(chunk_id::Int, chunksize::Int, n::Int) =
    ((chunk_id - 1) * chunksize + 1) : min(chunk_id * chunksize, n)

# `f` is only handed on to `_tforeach`, which is the one case where Julia skips
# specializing on a function argument, so it takes the type parameter here. The
# `_tforeach` methods below call `f` and specialize without one.
function tforeach(f::F, iter, scheduler=DynamicScheduler(); kwargs...) where {F}
    if Threads.nthreads() > 1
        _tforeach(f, iter, get_scheduler(scheduler); kwargs...)
    else
        _tforeach(f, iter, SequentialScheduler(); kwargs...)
    end
end

# The single point to change for a custom threading backend.
function _tforeach(f, iter, ::StaticScheduler)
    Threads.@threads :static for i in iter
        @inline f(i)
    end
end
function _tforeach(f, iter, ::DynamicScheduler)
    Threads.@threads :dynamic for i in iter
        @inline f(i)
    end
end
function _tforeach(f, iter, ::GreedyScheduler)
    Threads.@threads :greedy for i in iter
        @inline f(i)
    end
end
function _tforeach(f, iter, ::SequentialScheduler)
    for i in iter
        @inline f(i)
    end
end

"""
    @threaded [scheduler] for ...
    @threaded [scheduler] @P2G ...
    @threaded @P2G ...

A macro similar to `Threads.@threads`, but also works with
[`@P2G`](@ref), [`@G2P`](@ref), [`@G2P2G`](@ref), and [`@P2G_Matrix`](@ref) macros for particle-grid transfers.

The optional `scheduler` can be `:static`, `:dynamic`, `:greedy`, or `:nothing`
(sequential execution). The default is `:dynamic`.

What the three parallel schedulers select depends on what is being threaded. On
a plain loop, and on [`@G2P`](@ref), they pick the corresponding
`Threads.@threads` variant. On a partitioned transfer -- [`@P2G`](@ref),
[`@G2P2G`](@ref) and [`@P2G_Matrix`](@ref) given a [`ThreadPartition`](@ref) --
they instead pick how each color group is divided among the workers: `:static`
by region count, `:dynamic` by particle count, and `:greedy` one region at a
time on demand, which balances best and is several times slower because it
scatters each worker's grid writes.

See also [`ThreadPartition`](@ref).

!!! note
    If multi-threading is disabled or only one thread is available,
    this macro falls back to sequential execution.

# Examples
```julia
# Parallel loop
@threaded for i in 1:100
    println(i)
end

# Grid-to-particle transfer
@threaded @G2P grid=>i particles=>p weights=>ip begin
    v[p] = @∑ w[ip] * v[i]
end
```
"""
macro threaded(expr)
    threaded_expr(QuoteNode(:dynamic), expr)
end

macro threaded(schedule::QuoteNode, expr)
    threaded_expr(schedule, expr)
end

function threaded_expr(schedule::QuoteNode, expr::Expr)
    if Meta.isexpr(expr, :for)
        head = expr.args[1]
        index = esc(head.args[1])
        iter = esc(head.args[2])
        body = esc(expr.args[2])
        quote
            Tesserae.tforeach($iter, $schedule) do $index
                $body
            end
        end
    elseif Meta.isexpr(expr, :macrocall) &&
           (expr.args[1] in (Symbol("@P2G"), Symbol("@G2P"), Symbol("@G2P2G"), Symbol("@P2G_Matrix"), Symbol("@foreach")))
        insert!(expr.args, 3, schedule)
        esc(expr)
    else
        error("wrong usage for @threaded")
    end
end

# C = Aᵀ B α + C β
function tmul!(C::StridedVecOrMat{T}, A::SparseMatrixCSC{T}, B::StridedVecOrMat{T}, α, β) where {T <: Real}
    rows = rowvals(A)
    vals = nonzeros(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k in 1:size(C, 2)
        @threaded for j in 1:size(A, 2)
            @inbounds begin
                tmp = zero(T)
                for i in nzrange(A, j)
                    row = rows[i]
                    val = vals[i]
                    tmp += val * B[row, k]
                end
                C[j, k] += tmp * α
            end
        end
    end
    C
end
tmul!(C::StridedVecOrMat, A::SparseMatrixCSC, B::StridedVecOrMat) = tmul!(C, A, B, true, false)
