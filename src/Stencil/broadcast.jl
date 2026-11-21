import Base.Broadcast

struct CommonSource{A}
    parent::A
end

"""
    common(A)

Use the array `A` in common-range broadcast updates. Indexing the returned
adapter with a Region defers that reference until the destination and all
inputs have been restricted together to positions where every read exists.
Each assignment chooses its own intersection; other positions stay unchanged.

```julia
using Tesserae.Stencil: common

f, temperature = common(flux), common(values)
@views @. f[F] = -(temperature[F+e/2] - temperature[F-e/2]) / h
```

Ordinary `Region` views retain their usual bounds checks. `@stencil` applies this adapter automatically.
"""
common(A::AbstractArray) = CommonSource(A)
common(A::CommonSource) = A
Base.firstindex(A::CommonSource, d...) = firstindex(A.parent, d...)
Base.lastindex(A::CommonSource, d...) = lastindex(A.parent, d...)

# A deferred broadcast operand, not an AbstractArray pretending all of its
# candidate indices can already be read. No field values are read here.
struct RegionOperand{A, N, T, F<:GridFrame{N, T}}
    parent::A
    indices::NTuple{N, IndexRange}
    frame::F
end
function RegionOperand(A::AbstractArray{T, N}, r::Region{N}) where {T, N}
    RegionOperand(A, Base.to_indices(A, (r,)), r.frame)
end

function RegionOperand(A::AbstractArray{T, N}, regions::AbstractVector{<:Region{N}}) where {T, N}
    indices = Base.to_indices(A, (regions,))
    RegionOperand(A, indices, first(regions).frame)
end

# Dispatch on our adapter only. Parent array types keep ownership of their
# view methods; their native views are created after resolving the range.
Base.getindex(A::CommonSource, r::Union{Region, AbstractVector{<:Region}}) = RegionOperand(A.parent, r)
Base.getindex(A::CommonSource, indices...) = Base.maybeview(A.parent, indices...)
Base.view(A::CommonSource, r::Union{Region, AbstractVector{<:Region}}) = A[r]
Base.view(A::CommonSource, indices...) = view(A.parent, indices...)
Base.size(v::RegionOperand) = map(length, v.indices)
Base.ndims(::RegionOperand{A, N}) where {A, N} = N
Base.axes(v::RegionOperand) = map(Base.OneTo, size(v))
Base.size(v::RegionOperand, d::Integer) = d > ndims(v) ? 1 : size(v)[d]
Base.axes(v::RegionOperand, d::Integer) = Base.OneTo(size(v, d))
Base.eltype(::Type{<:RegionOperand{A}}) where {A} = eltype(A)
Base.broadcastable(v::RegionOperand) = v

# Keep expressions deferred through blocks and macros until the destination
# resolves their common range. Native array styles take over after lowering.
struct CommonStyle <: Broadcast.BroadcastStyle end
Broadcast.BroadcastStyle(::Type{<:RegionOperand}) = CommonStyle()
Broadcast.BroadcastStyle(::CommonStyle, ::Broadcast.BroadcastStyle) = CommonStyle()
Broadcast.BroadcastStyle(::CommonStyle, ::CommonStyle) = CommonStyle()
Broadcast.BroadcastStyle(::CommonStyle, ::Broadcast.Unknown) = CommonStyle()
Broadcast.materialize(bc::Broadcast.Broadcasted{CommonStyle}) = bc

function Base.getindex(::RegionOperand, indices...)
    throw(ArgumentError("common-region inputs require an in-place broadcast with a common-region destination"))
end

function available(v::RegionOperand)
    map(v.indices, axes(v.parent)) do r, ax
        span = findall(in(ax), r)
        first(span):last(span)
    end
end

# Work in local broadcast coordinates, not absolute physical coordinates.
# This also permits paired reads from opposite periodic boundaries.
# Keep local bounds and slices as UnitRanges; physical selections retain their step.
function restrict(bounds, v::RegionOperand)
    valid = available(v)
    ntuple(length(bounds)) do d
        d > ndims(v) && return bounds[d]
        if size(v)[d] == 1
            isempty(valid[d]) ? (1:0) : bounds[d]
        else
            intersect(bounds[d], valid[d])
        end
    end
end
restrict(bounds, bc::Broadcast.Broadcasted) = foldl(restrict, bc.args; init = bounds)
restrict(bounds, x) = bounds

function localslices(x, bounds)
    ntuple(ndims(x)) do d
        axis = axes(x, d)
        d > length(bounds) && return first(axis):last(axis)
        isempty(bounds[d]) ? (1:0) : size(x)[d] == 1 ? (first(axis):last(axis)) : bounds[d]
    end
end
function lower(v::RegionOperand, bounds)
    indices = map(getindex, v.indices, localslices(v, bounds))
    !Base.has_offset_axes(v.parent) && indices == axes(v.parent) ? v.parent : native_view(v.parent, indices)
end
lower(x::AbstractArray, bounds) = view(x, localslices(x, bounds)...)
lower(x::Tuple, bounds) = length(x) == 1 ? x : x[bounds[1]]
lower(x, bounds) = x
function lower(bc::Broadcast.Broadcasted, bounds)
    Broadcast.broadcasted(bc.f, map(x -> lower(x, bounds), bc.args)...)
end

checkframes(frame, x) = nothing
checkframes(frame, v::RegionOperand) = checkframe(frame, v.frame)
checkframes(frame, bc::Broadcast.Broadcasted) = foreach(x -> checkframes(frame, x), bc.args)

# Specialize Base's broadcast kernel after resolving the native array types.
@noinline materialize_native!(dest, expression) = Broadcast.materialize!(dest, expression)

function copy_common!(dest::RegionOperand, bc::Broadcast.Broadcasted)
    checkframes(dest.frame, bc)
    # Check ordinary broadcasting compatibility before cropping anything.
    checked = Broadcast.instantiate(Broadcast.Broadcasted(bc.f, bc.args, axes(dest)))
    bounds = restrict(available(dest), checked)
    native_dest = lower(dest, bounds)
    any(isempty, bounds) && return native_dest
    native_bc = lower(checked, bounds)
    # Native broadcast owns numerical execution and its normal alias handling.
    materialize_native!(native_dest, native_bc)
    native_dest
end

# Once the ranges are resolved, numerical backend styles are selected again
# from the ordinary native views. Common regions support in-place dot expressions.
Broadcast.materialize!(dest::RegionOperand, bc::Broadcast.Broadcasted) = copy_common!(dest, bc)
