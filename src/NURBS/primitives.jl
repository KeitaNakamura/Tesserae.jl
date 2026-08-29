"""
    line(p₀::Vec, p₁::Vec) -> ControlNet

Create a linear curve from `p₀` to `p₁`.
"""
function line(p0::Vec{dim}, p1::Vec{dim}) where {dim}
    points = [p0, p1]
    axis = open_bspline_axis(eltype(eltype(points)), linear, 1)
    ControlNet((axis,), points)
end

"""
    polyline(points::AbstractVector{<: Vec}) -> ControlNet

Create a piecewise-linear curve through the given points.
"""
function polyline(points::AbstractVector{Vec{dim, T}}) where {dim, T}
    length(points) ≥ 2 || throw(ArgumentError("polyline needs at least two points"))
    axis = open_bspline_axis(T, linear, length(points) - 1)
    ControlNet((axis,), collect(points))
end

"""
    circle(center::Vec{2}, radius::Real) -> ControlNet
    circle(center::Vec{3}, radius::Real; normal::Vec{3}=Vec(0,0,1), xaxis::Vec{3}=default_arc_xaxis(normal)) -> ControlNet

Create an exact rational quadratic circle.
"""
function circle(center::Vec{2, T}, radius::Real) where {T}
    arc(center, radius, zero(T), T(2π))
end

function circle(center::Vec{3, T}, radius::Real; normal::Vec{3}=Vec(0,0,1), xaxis=default_arc_xaxis(normal)) where {T}
    arc(center, radius, zero(T), T(2π); normal, xaxis)
end

"""
    arcunit(θ::Real) -> ControlNet

Create an exact rational quadratic circular arc centered at the origin, with
radius one, start angle zero, and sweep angle `θ`.
"""
function arcunit(θ::Real)
    nsegs = arc_nsegments(θ)
    Δθ = θ / nsegs
    axis = arc_bspline_axis(typeof(Δθ), nsegs)
    points, weights = arcunit_points_weights(nsegs, Δθ, cos(Δθ/2))
    ControlNet((axis,), points, weights)
end

function arc_bspline_axis(::Type{T}, nsegs::Integer) where {T}
    knots = Vector{T}(undef, 2nsegs + 4)
    knots[1:3] .= 0
    knots[end-2:end] .= 1
    for i in 4:(lastindex(knots)-3)
        knots[i] = T((i-2) ÷ 2) / nsegs
    end
    BSplineAxis(quadratic, knots)
end

function arc_nsegments(θ::Real)
    quarter_turns = 2abs(θ) / π
    0 < quarter_turns ≤ 4 || throw(ArgumentError("arc angle must be in (0, 2π]"))
    ceil(Int, quarter_turns)
end

function arcunit_points_weights(nsegs::Integer, Δθ::T, w_mid::T) where {T}
    points  = Vector{Vec{2, T}}(undef, 2nsegs + 1)
    weights = Vector{T}(undef, 2nsegs + 1)
    for i in eachindex(points)
        θ = (i-1) * Δθ / 2
        weights[i] = ifelse(isodd(i), one(w_mid), w_mid)
        points[i] = arcunit_point(θ) / weights[i]
    end
    points, weights
end
arcunit_point(Δθ) = Vec(cos(Δθ), sin(Δθ))

"""
    arc(center::Vec{2}, radius::Real, θ₀::Real, θ₁::Real) -> ControlNet
    arc(center::Vec{3}, radius::Real, θ₀::Real, θ₁::Real; normal::Vec{3}=Vec(0,0,1), xaxis::Vec{3}=default_arc_xaxis(normal)) -> ControlNet

Create an exact rational quadratic circular arc. The 2D form lies in the
global x-y plane. The 3D form uses `normal` as the plane normal; `θ₀` and `θ₁`
are measured from `xaxis` in that plane.
"""
function arc(center::Vec{2}, radius::Real, θ₀::Real, θ₁::Real)
    arc_from_axes(center, radius, θ₀, θ₁, Vec(1,0), Vec(0,1))
end

function arc(center::Vec{3}, radius::Real, θ₀::Real, θ₁::Real; normal::Vec{3}=Vec(0,0,1), xaxis=default_arc_xaxis(normal))
    n_norm = norm(normal)
    iszero(n_norm) && throw(ArgumentError("arc normal must be nonzero"))
    n = normal / n_norm
    x = normalize_arc_xaxis(n, xaxis)
    arc_from_axes(center, radius, θ₀, θ₁, x, n × x)
end

function arc_from_axes(center::Vec{dim}, radius::Real, θ₀::Real, θ₁::Real, xaxis::Vec{dim}, yaxis::Vec{dim}) where {dim}
    radius > 0 || throw(ArgumentError("arc radius must be positive"))
    θ = θ₁ - θ₀
    0 < abs(θ)/2 ≤ π || throw(ArgumentError("arc angle must be in (0, 2π]"))
    unit = arcunit(θ)
    x =  cos(θ₀) * xaxis + sin(θ₀) * yaxis
    y = -sin(θ₀) * xaxis + cos(θ₀) * yaxis
    points = map(unit.points) do point
        center + radius * (point[1] * x + point[2] * y)
    end
    ControlNet(unit.axes, points, unit.weights)
end

"""
    default_arc_xaxis(normal::Vec{3}) -> Vec{3}

Choose the default zero-angle direction for a 3D arc from its plane normal,
following the AutoCAD/DXF object coordinate system convention.
"""
function default_arc_xaxis(normal::Vec{3})
    if abs(normal[1]) < 1/64 && abs(normal[2]) < 1/64
        Vec(0,1,0) × normal
    else
        Vec(0,0,1) × normal
    end
end

function normalize_arc_xaxis(n::Vec{3}, xaxis::Vec{3})
    projected = xaxis - (xaxis ⋅ n) * n
    l = norm(projected)
    l > 0 || throw(ArgumentError("arc xaxis must not be parallel to normal"))
    projected / l
end
