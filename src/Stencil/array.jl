# padded to construct padded arrays
function padded(A::AbstractArray; pad::Int)
    B = similar(A, size(A) .+ 2*pad)
    inner(B; pad) .= A
    return B
end

struct CatDimView{T, N, TA <: AbstractArray{T, N}} <: AbstractArray{T, N}
    first::TA
    last::TA
    d::Int
    function CatDimView{T, N, TA}(first::TA, last::TA, d::Int) where {T, N, TA}
        @assert 1 ≤ d ≤ N
        @assert all(j -> j==d ? true : size(first, j) == size(last, j), 1:N)
        new{T, N, TA}(first, last, d)
    end
end
function CatDimView(first::TA, last::TA, d::Int) where {T, N, TA <: AbstractArray{T, N}}
    CatDimView{T, N, TA}(first, last, d)
end

function Base.size(x::CatDimView{<: Any, N}) where {N}
    (; first, last, d) = x
    ntuple(Val(N)) do j
        j == d ? size(first, j) + size(last, j) : size(first, j)
    end
end
Base.IndexStyle(::Type{<: CatDimView}) = IndexCartesian()

@inline function Base.getindex(x::CatDimView{<: Any, N}, I::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(x, I...)
    @inbounds begin
        (; first, last, d) = x
        if I[d] ≤ size(first, d)
            return first[I...]
        else
            J = ntuple(j -> j==d ? I[j] - size(first, j) : I[j], Val(N))
            return last[J...]
        end
    end
end

abstract type Location end

struct Cell <: Location end
struct Face <: Location
    axis::Int
end
getaxis(x::Face) = x.axis

struct StencilArray{Loc <: Location, T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    loc::Loc
    parent::A
end

Base.parent(x::StencilArray) = x.parent
Base.size(x::StencilArray) = size(parent(x))
Base.axes(x::StencilArray) = axes(parent(x))
Base.IndexStyle(::Type{StencilArray{F, T, N, A}}) where {F, T, N, A} = IndexStyle(A)
@inline function Base.getindex(x::StencilArray, I...)
    @boundscheck checkbounds(parent(x), I...)
    @inbounds parent(x)[I...]
end
@inline function Base.setindex!(x::StencilArray, v, I...)
    @boundscheck checkbounds(parent(x), I...)
    @inbounds parent(x)[I...] = v
    x
end

Base.Broadcast.broadcastable(A::StencilArray) = parent(A)
Base.view(A::StencilArray, I...) = view(parent(A), I...)

getlocation(x::StencilArray) = x.loc
getaxis(x::StencilArray{Face}) = getaxis(getlocation(x))

# interior_indices
function interior_indices(::Cell, axes::NTuple{N, AbstractUnitRange}; pad::Int) where {N}
    ranges = map(axes) do ax
        first(ax) + pad : last(ax) - pad
    end
    indices = CartesianIndices(ranges)
    @assert !isempty(indices)
    return indices
end

function interior_indices(loc::Face, axes::NTuple{N,AbstractUnitRange}; pad::Int, dropboundary::Bool=false) where {N}
    a = getaxis(loc)
    ranges = ntuple(Val(N)) do d
        ax = axes[d]
        extra = ifelse(dropboundary && d == a, 1, 0)
        (first(ax) + pad + extra) : (last(ax) - pad - extra)
    end
    indices = CartesianIndices(ranges)
    @assert !isempty(indices)
    return indices
end

function offaxis_range(ax::AbstractUnitRange, pad::Int, trim::Bool)
    range = trim ? ((first(ax)+pad):(last(ax)-pad)) : ax
    @assert first(range) ≤ last(range)
    return range
end

innerfront_axis(::Cell, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = first(ax)+pad:first(ax)+pad+n-1
outerfront_axis(::Cell, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = first(ax)+pad-n:first(ax)+pad-1
innerback_axis( ::Cell, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = last(ax)-pad-n+1:last(ax)-pad
outerback_axis( ::Cell, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = last(ax)-n+1:last(ax)

innerfront_axis(loc::Face, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = d == getaxis(loc) ? (first(ax)+pad : first(ax)+pad+n) : innerfront_axis(Cell(), ax, d, n, pad)
outerfront_axis(loc::Face, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = d == getaxis(loc) ? (first(ax)+pad-n : first(ax)+pad) : outerfront_axis(Cell(), ax, d, n, pad)
innerback_axis(loc::Face, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = d == getaxis(loc) ? (last(ax)-pad-n : last(ax)-pad) : innerback_axis(Cell(), ax, d, n, pad)
outerback_axis(loc::Face, ax::AbstractUnitRange, d::Int, n::Int, pad::Int) = d == getaxis(loc) ? (last(ax)-pad : last(ax)-pad+n) : outerback_axis(Cell(), ax, d, n, pad)

# Build an N-dimensional CartesianIndices region where axis `d` uses the explicit `range`,
# and every other axis uses `offaxis_range(axes[j], pad, trim)` (i.e. either the full axis
# or the axis trimmed by `pad`, depending on `trim`).
function _build_indices(range::AbstractUnitRange, axes::NTuple{N, AbstractUnitRange}, d::Int, pad::Int, trim::Bool) where {N}
    ax = axes[d]
    @assert first(ax) ≤ first(range) ≤ last(range) ≤ last(ax)
    CartesianIndices(ntuple(Val(N)) do j
        j == d ? range : offaxis_range(axes[j], pad, trim)
    end)
end

function innerfront_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int, n::Int; pad::Int, trim::Bool=false) where {N}
    @assert 1 ≤ d ≤ N
    @assert pad ≥ 0 && n ≥ 0
    _build_indices(innerfront_axis(loc, axes[d], d, n, pad), axes, d, pad, trim)
end
innerfront_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int; pad::Int, trim::Bool=false) where {N} = innerfront_indices(loc, axes, d, pad; pad, trim)

function outerfront_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int, n::Int; pad::Int, trim::Bool=false) where {N}
    @assert 1 ≤ d ≤ N
    @assert pad ≥ 0 && 0 ≤ n ≤ pad
    _build_indices(outerfront_axis(loc, axes[d], d, n, pad), axes, d, pad, trim)
end
outerfront_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int; pad::Int, trim::Bool=false) where {N} = outerfront_indices(loc, axes, d, pad; pad, trim)

function innerback_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int, n::Int; pad::Int, trim::Bool=false) where {N}
    @assert 1 ≤ d ≤ N
    @assert pad ≥ 0 && n ≥ 0
    _build_indices(innerback_axis(loc, axes[d], d, n, pad), axes, d, pad, trim)
end
innerback_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int; pad::Int, trim::Bool=false) where {N} = innerback_indices(loc, axes, d, pad; pad, trim)

function outerback_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int, n::Int; pad::Int, trim::Bool=false) where {N}
    @assert 1 ≤ d ≤ N
    @assert pad ≥ 0 && 0 ≤ n ≤ pad
    _build_indices(outerback_axis(loc, axes[d], d, n, pad), axes, d, pad, trim)
end
outerback_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int; pad::Int, trim::Bool=false) where {N} = outerback_indices(loc, axes, d, pad; pad, trim)

function outer_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int, n::Int; pad::Int, trim::Bool=false) where {N}
    front = outerfront_indices(loc, axes, d, n; pad, trim)
    back = outerback_indices(loc, axes, d, n; pad, trim)
    CatDimView(front, back, d)
end
outer_indices(loc::Location, axes::NTuple{N, AbstractUnitRange}, d::Int; pad::Int, trim::Bool=false) where {N} = outer_indices(loc, axes, d, pad; pad, trim)

# inner/outer
inner(A::StencilArray; kwargs...) = view(A, interior_indices(getlocation(A), axes(A); kwargs...))
outer(A::StencilArray, args...; kwargs...) = view(A, outer_indices(getlocation(A), axes(A), args...; kwargs...))
# innerfront/outerfront
innerfront(A::StencilArray, args...; kwargs...) = view(A, innerfront_indices(getlocation(A), axes(A), args...; kwargs...))
outerfront(A::StencilArray, args...; kwargs...) = view(A, outerfront_indices(getlocation(A), axes(A), args...; kwargs...))
# innerback/outerback
innerback(A::StencilArray, args...; kwargs...) = view(A, innerback_indices(getlocation(A), axes(A), args...; kwargs...))
outerback(A::StencilArray, args...; kwargs...) = view(A, outerback_indices(getlocation(A), axes(A), args...; kwargs...))

function mirror(x, axis::Int)
    axes = parentindices(x)
    inds = ntuple(d -> d == axis ? reverse(axes[d]) : axes[d], Val(ndims(x)))
    return view(parent(x), inds...)
end

function foldpadfront!(f, dest::StencilArray{Loc}, axis::Int, srcs::StencilArray{Loc}...; pad::Int) where {Loc}
    IF(A) = innerfront(A, axis; pad, trim=true)
    OF(A) = mirror(outerfront(A, axis; pad, trim=true), axis)
    map!((y, xs...) -> y + f(xs...), IF(dest), IF(dest), map(OF, srcs)...)
    return dest
end

function foldpadback!(f, dest::StencilArray{Loc}, axis::Int, srcs::StencilArray{Loc}...; pad::Int) where {Loc}
    IB(A) = innerback(A, axis; pad, trim=true)
    OB(A) = mirror(outerback(A, axis; pad, trim=true), axis)
    map!((y, xs...) -> y + f(xs...), IB(dest), IB(dest), map(OB, srcs)...)
    return dest
end

# flip shortcut (self-fold)
foldpadfront!(A::StencilArray, axis::Int; pad::Int, flip::Bool=false) = foldpadfront!(ifelse(flip, -, +), A, axis, A; pad)
foldpadback!(A::StencilArray, axis::Int; pad::Int, flip::Bool=false) = foldpadback!(ifelse(flip, -, +), A, axis, A; pad)
foldpad!(A::StencilArray, axis::Int; pad::Int, flip::Bool=false) = (foldpadfront!(A, axis; pad, flip); foldpadback!(A, axis; pad, flip))
function foldpad!(A::StencilArray; pad::Int, flip::AbstractVector{Bool}=falses(ndims(A)))
    @assert length(flip) == ndims(A)
    for axis in 1:ndims(A)
        foldpad!(A, axis; pad, flip=flip[axis])
    end
    return A
end

function mirrorpadfront!(f, dest::StencilArray, axis::Int; pad::Int)
    IF(A) = mirror(innerfront(A, axis; pad, trim=false), axis)
    OF(A) = outerfront(A, axis; pad, trim=false)
    map!(f, OF(dest), IF(dest))
    return dest
end

function mirrorpadback!(f, dest::StencilArray, axis::Int; pad::Int)
    IB(A) = mirror(innerback(A, axis; pad, trim=false), axis)
    OB(A) = outerback(A, axis; pad, trim=false)
    map!(f, OB(dest), IB(dest))
    return dest
end

# flip shortcut
mirrorpadfront!(A::StencilArray, axis::Int; pad::Int, flip::Bool=false) = mirrorpadfront!(ifelse(flip, -, +), A, axis; pad)
mirrorpadback!(A::StencilArray, axis::Int; pad::Int, flip::Bool=false) = mirrorpadback!(ifelse(flip, -, +), A, axis; pad)
mirrorpad!(A::StencilArray, axis::Int; pad::Int, flip::Bool=false) = (mirrorpadfront!(A, axis; pad, flip); mirrorpadback!(A, axis; pad, flip))
function mirrorpad!(A::StencilArray; pad::Int, flip::AbstractVector{Bool}=falses(ndims(A)))
    @assert length(flip) == ndims(A)
    for axis in 1:ndims(A)
        mirrorpad!(A, axis; pad, flip=flip[axis])
    end
    return A
end

# extendsize
function extendsize(dims::NTuple{N}, ax::Int, n::Int) where {N}
    ntuple(d -> d == ax ? dims[d]+n : dims[d], Val(N))
end

# infersize
infersize(destloc::Cell, srcloc::Cell, srcdims::Dims) = srcdims
function infersize(destloc::Face, srcloc::Cell, srcdims::Dims)
    extendsize(srcdims, getaxis(destloc), 1)
end
function infersize(destloc::Cell, srcloc::Face, srcdims::Dims)
    extendsize(srcdims, getaxis(srcloc), -1)
end
function infersize(destloc::Face, srcloc::Face, srcdims::Dims)
    @assert getaxis(destloc) == getaxis(srcloc)
    return srcdims
end

# check_size
function check_size(dest::StencilArray, src::StencilArray)
    @assert size(dest) == infersize(getlocation(dest), getlocation(src), size(src))
end

# similar for `Location`
function Base.similar(src::StencilArray, ::Type{T}, destloc::Location) where {T}
    destdims = infersize(destloc, getlocation(src), size(src))
    StencilArray(destloc, similar(parent(src), T, destdims))
end
function Base.similar(src::StencilArray, destloc::Location)
    similar(src, eltype(src), destloc)
end
function Base.similar(src::StencilArray)
    similar(src, eltype(src), getlocation(src))
end
