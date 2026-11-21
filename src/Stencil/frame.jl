# Frames describe the coordinate basis, independently of a field's placement.
struct GridFrame{N, T, A}
    origin::NTuple{N, T}
    spacing::T
    axes::A
end

GridFrame(::Val{N}) where {N} = GridFrame(ntuple(_ -> 0.0, Val(N)), 1.0, nothing)
GridFrame(mesh::CartesianMesh) = GridFrame(map(first, mesh.axes), spacing(mesh), mesh.axes)

function Base.:(==)(a::GridFrame, b::GridFrame)
    a.origin == b.origin && a.spacing == b.spacing && samebasis(a, b)
end
function Base.isequal(a::GridFrame, b::GridFrame)
    isequal(a.origin, b.origin) && isequal(a.spacing, b.spacing) && samebasis(a, b)
end
Base.hash(frame::GridFrame, h::UInt) = hash((!isnothing(frame.axes), frame.origin, frame.spacing), h)

function samebasis(a::GridFrame, b::GridFrame)
    isnothing(a.axes) == isnothing(b.axes) || return false
    a.axes === b.axes && return true
    all(eachindex(a.origin)) do d
        indices = 1:max(length(a.axes[d]), length(b.axes[d]))
        all((true, false)) do aligned
            xs = coordinate_axis(indices, aligned, a, d)
            ys = coordinate_axis(indices, aligned, b, d)
            all(x == y for (x, y) in zip(xs, ys))
        end
    end
end

function checkframe(a::GridFrame, b::GridFrame)
    a == b || throw(ArgumentError("geometry and regions must use the same grid coordinate frame"))
    nothing
end

# Julia's range coefficients preserve its floating-point rounding without
# converting every coordinate to a rational number.
gridbasis(origin, h) = (origin, h)
function gridbasis(origin::T, h::T) where {T<:AbstractFloat}
    axis = range(origin; step=h, length=1)
    axis.ref, axis.step
end

function coordinate_axis(indices, aligned::Bool, h::Real, origin::Real, axis=nothing)
    x0, hd = promote(origin, h)
    coordinate_axis(indices, aligned, gridbasis(x0, hd), typeof(x0+hd/2), axis)
end

function coordinate_axis(indices, aligned::Bool, basis::Tuple, ::Type{T}, axis) where {T}
    ref, step = basis
    start = aligned ? ref : ref+step/2
    source_indices = aligned && axis !== nothing ? (firstindex(axis):lastindex(axis)) : (1:0)
    (gridcoordinate(axis, (start, step), i, source_indices, T) for i in indices)
end

function coordinate_axis(indices, aligned::Bool, frame::GridFrame, d::Int)
    axis = frame.axes === nothing ? nothing : frame.axes[d]
    coordinate_axis(indices, aligned, frame.spacing, frame.origin[d], axis)
end

# Preserve supplied grid points; evaluate centers and halos from the prepared basis.
function gridcoordinate(axis, basis::Tuple, i::Integer, source_indices, ::Type{T}) where {T}
    if i in source_indices
        return T(@inbounds axis[i])
    end
    T(basis[1]+(i-1)*basis[2])
end

function lattice_integer(x::Real)
    isfinite(x) && isinteger(x) || throw(ArgumentError("the transformation does not map to the half-cell lattice"))
    Int(x)
end
