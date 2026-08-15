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

################
# Order/Degree #
################

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

#############
# Utilities #
#############

nfill(v, ::Val{dim}) where {dim} = ntuple(i->v, Val(dim))

# zero_recursive
@generated function zero_recursive(::Type{T}) where {T}
    isbitstype(T) || return :(throw(ArgumentError("`zero_recursive` supports only `isbitstype`, got $T")))
    :(@_inline_meta; zero(T))
end
@generated function zero_recursive(::Type{T}) where {T <: Union{Tuple, NamedTuple}}
    exps = [:(zero_recursive($t)) for t in fieldtypes(T)]
    :(@_inline_meta; T(($(exps...),)))
end
zero_recursive(x) = zero_recursive(typeof(x))

# fillzero!
function fillzero!(x::AbstractArray)
    fill!(x, zero_recursive(eltype(x)))
    x
end

# `fill!` stores a composite element such as `Vec{2,Float64}` one field at a
# time, which measures about half the speed of a memset over the same bytes --
# and grid fields are mostly composite. Zeroing a dense array can go through
# memset instead whenever a zero of the element type is all-zero bytes.
#
# The test is structural -- every leaf field is a number or a `Bool`, whose zero
# is all-zero bytes -- rather than an inspection of a zero value, since reading
# the bytes of one would also read padding, which is undefined. A type built
# only from numbers but with a `zero` that is not zero would take this path
# wrongly; `zero_recursive` reaches leaves the same way, so such a type is
# already outside what `fillzero!` promises. Padding itself is fine: memset
# zeroes it and nothing reads it.
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

# fastsum
@inline function fastsum(f, iter)
    ret = zero(Base._return_type(f, Tuple{eltype(iter)}))
    @simd for x in iter
        ret += @inline f(x)
    end
    ret
end

# commas
commas(num::Integer) = replace(string(num), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")

# getx
getx(x) = getproperty(x, first(propertynames(x)))
getx(x::Vec) = x
getx(x::Vector{<: Vec}) = x

# dropat
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

############
# MapArray #
############

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

#########
# Trues #
#########

struct Trues{N} <: AbstractArray{Bool, N}
    dims::Dims{N}
end
Base.size(t::Trues) = t.dims
Base.IndexStyle(::Type{<: Trues}) = IndexLinear()
@inline function Base.getindex(t::Trues, i::Integer)
    @boundscheck checkbounds(t, i)
    true
end

##################
# threaded macro #
##################

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

# `f` is only handed on to `_tforeach`, which is the one case where Julia skips
# specializing on a function argument, so it takes the type parameter here. The
# `_tforeach` methods below call `f` and specialize without one.
# Chunk `chunk_id` of `n` items split into pieces of `chunksize`. Empty when the
# last chunks have nothing left, which happens whenever `n` is not a multiple.
@inline chunk_range(chunk_id::Int, chunksize::Int, n::Int) =
    ((chunk_id - 1) * chunksize + 1) : min(chunk_id * chunksize, n)

function tforeach(f::F, iter, scheduler=DynamicScheduler(); kwargs...) where {F}
    if Threads.nthreads() > 1
        _tforeach(f, iter, get_scheduler(scheduler); kwargs...)
    else
        _tforeach(f, iter, SequentialScheduler(); kwargs...)
    end
end

# Modify the following funcitons for custom multi-threading. For now, just use Threads.@threads.
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

#########
# tmul! #
#########

# multithreading C = Aᵀ B α + C β
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
