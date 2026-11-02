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
f, temperature = common(flux), common(values)
@views @. f[F] = -(temperature[F+e/2] - temperature[F-e/2]) / h
```

The assignment returns the native view that was updated. Ordinary `Region`
indexing remains strict. `@stencil` applies this adapter automatically.
"""
common(A::AbstractArray) = CommonSource(A)
common(A::CommonSource) = A
Base.firstindex(A::CommonSource, d...) = firstindex(A.parent, d...)
Base.lastindex(A::CommonSource, d...) = lastindex(A.parent, d...)

# A deferred broadcast operand, not an AbstractArray pretending all of its
# candidate indices can already be read. No field values are read here.
struct RegionOperand{T,N,A,S}
    parent::A
    selector::S
end
function RegionOperand(A::AbstractArray{T,N}, r::Union{Region{N},ShiftedRegion{N}}) where {T,N}
    RegionOperand{T,N,typeof(A),typeof(r)}(A, r)
end

# Dispatch on our adapter only. Parent array types keep ownership of their
# view methods; their native views are created after resolving the range.
Base.getindex(A::CommonSource, r::Union{Region,ShiftedRegion}) = RegionOperand(A.parent, r)
Base.getindex(A::CommonSource, indices...) = Base.maybeview(A.parent, indices...)
Base.view(A::CommonSource, r::Union{Region,ShiftedRegion}) = A[r]
Base.view(A::CommonSource, indices...) = view(A.parent, indices...)
Base.size(v::RegionOperand) = size(v.selector)
Base.ndims(::RegionOperand{T,N}) where {T,N} = N
Base.axes(v::RegionOperand) = map(Base.OneTo, size(v))
Base.size(v::RegionOperand, d::Integer) = d > ndims(v) ? 1 : size(v)[d]
Base.axes(v::RegionOperand, d::Integer) = Base.OneTo(size(v, d))
Base.eltype(::Type{<:RegionOperand{T}}) where {T} = T
Base.broadcastable(v::RegionOperand) = v
Broadcast.BroadcastStyle(::Type{<:RegionOperand{T,N}}) where {T,N} = Broadcast.DefaultArrayStyle{N}()

readranges(v::RegionOperand) = Base.to_indices(v.parent, (v.selector,))
function Base.getindex(::RegionOperand, indices...)
    throw(ArgumentError("common-region inputs require an in-place broadcast with a common-region destination"))
end

intersectrange(a, b) = max(first(a), first(b)):min(last(a), last(b))
function available(v::RegionOperand{T,N}) where {T,N}
    requested = readranges(v)
    ntuple(N) do d
        axis = axes(v.parent, d)
        lo = max(1, first(axis)-first(requested[d])+1)
        hi = min(size(v)[d], last(axis)-first(requested[d])+1)
        lo <= hi ? (lo:hi) : (1:0)
    end
end

# Work in local broadcast coordinates, not absolute physical coordinates.
# This also permits paired reads from opposite periodic boundaries.
function restrict(bounds, v::RegionOperand)
    valid = available(v)
    ntuple(length(bounds)) do d
        d > ndims(v) && return bounds[d]
        if size(v)[d] == 1
            isempty(valid[d]) ? (1:0) : bounds[d]
        else
            intersectrange(bounds[d], valid[d])
        end
    end
end
restrict(bounds, bc::Broadcast.Broadcasted) = foldl(restrict, bc.args; init = bounds)
restrict(bounds, x) = bounds

function localslices(x, bounds)
    ntuple(ndims(x)) do d
        size(x)[d] == 1 ? axes(x, d) : bounds[d]
    end
end
function lower(v::RegionOperand, bounds)
    requested = readranges(v)
    local_ranges = localslices(v, bounds)
    indices = ntuple(ndims(v)) do d
        isempty(local_ranges[d]) ? (first(axes(v.parent, d)):(first(axes(v.parent, d))-1)) :
            requested[d][local_ranges[d]]
    end
    view(v.parent, indices...)
end
lower(x::AbstractArray, bounds) = view(x, localslices(x, bounds)...)
lower(x::Tuple, bounds) = length(x) == 1 ? x : x[bounds[1]]
lower(x, bounds) = x
function lower(bc::Broadcast.Broadcasted, bounds)
    Broadcast.broadcasted(bc.f, map(x -> lower(x, bounds), bc.args)...)
end

function copy_common!(dest::RegionOperand, bc::Broadcast.Broadcasted)
    # Check ordinary broadcasting compatibility before cropping anything.
    checked = Broadcast.instantiate(Broadcast.Broadcasted(bc.style, bc.f, bc.args, axes(dest)))
    bounds = restrict(available(dest), checked)
    native_dest = lower(dest, bounds)
    any(isempty, bounds) && return native_dest
    native_bc = lower(checked, bounds)
    # Native broadcast owns numerical execution and its normal alias handling.
    Broadcast.materialize!(native_dest, native_bc)
    native_dest
end

# A destination-specific entry point avoids imposing a new numerical broadcast
# style. Once the ranges are resolved, backend styles are selected again from
# the ordinary native views. Common regions support in-place dot expressions.
Broadcast.materialize!(dest::RegionOperand, bc::Broadcast.Broadcasted) = copy_common!(dest, bc)
