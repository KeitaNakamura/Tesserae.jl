"""
    loft(sections::AbstractVector{<: ControlNet}) -> ControlNet

Build one higher parametric dimension by stacking matching control nets.
"""
function loft(sections::AbstractVector{<: ControlNet{dim, pdim, T}}) where {dim, pdim, T}
    length(sections) ≥ 2 || throw(ArgumentError("loft requires at least two sections"))
    pdim < 3 || throw(ArgumentError("loft supports curves and surfaces"))

    reference = first(sections)
    for section in sections
        section.axes == reference.axes || throw(ArgumentError("section axes must match"))
        size(section.points) == size(reference.points) || throw(ArgumentError("section controlpoint sizes must match"))
    end

    # Lofting preserves each section as-is and uses a linear open knot vector
    # through the stacked sections in the new parametric direction.
    axis = open_bspline_axis(T, linear, length(sections) - 1)
    axes = (reference.axes..., axis)
    points, weights = loft_points_weights(sections)

    ControlNet(axes, points, weights)
end

function loft_points_weights(sections::AbstractVector{<: ControlNet{dim, pdim, T}}) where {dim, pdim, T}
    reference = first(sections)
    dims = (size(reference.points)..., length(sections))
    points = Array{Vec{dim, T}}(undef, dims)
    weights = Array{T}(undef, dims)
    for k in eachindex(sections), I in CartesianIndices(reference.points)
        points[I, k] = sections[k].points[I]
        weights[I, k] = sections[k].weights[I]
    end
    points, weights
end

"""
    sweep(net::ControlNet, direction::Vec; degree::Int=1, nspans::Int=1)

Sweep a control net by translating it along a straight vector.
"""
function sweep(net::ControlNet{dim, pdim, T}, direction::Vec{dim}; degree::Int=linear, nspans::Int=1) where {dim, pdim, T}
    pdim < 3 || throw(ArgumentError("sweep supports curves and surfaces"))
    degree < 1 && throw(ArgumentError("degree must be positive"))

    # A vector sweep adds one parametric direction. The new control points are
    # placed at the Greville stations of that added direction.
    axis = open_bspline_axis(T, degree, nspans)
    stations = greville_abscissae(axis)
    offsets = stations .* Ref(direction)
    points, weights = sweep_points_weights(net, offsets, ones(T, length(stations)))

    ControlNet((net.axes..., axis), points, weights)
end

"""
    sweep(section::ControlNet, trajectory::ControlNet)

Sweep a section along a trajectory curve by tensor-product translation.
"""
function sweep(section::ControlNet{dim, pdim}, trajectory::ControlNet{dim, 1}) where {dim, pdim}
    pdim < 3 || throw(ArgumentError("sweep supports curves and surfaces"))

    # This is the exact tensor-product sum A(u) + C(v). It does not rotate the
    # section frame, so no tangent-frame approximation is introduced here.
    points, weights = sweep_points_weights(section, trajectory.points, trajectory.weights)

    ControlNet((section.axes..., trajectory.axes...), points, weights)
end

function sweep_points_weights(section::ControlNet{dim, pdim, T}, offsets::AbstractVector{<: Vec{dim}}, offset_weights::AbstractVector) where {dim, pdim, T}
    dims = (size(section.points)..., length(offsets))
    points = Array{Vec{dim, T}}(undef, dims)
    weights = Array{T}(undef, dims)
    for k in eachindex(offsets), I in CartesianIndices(section.points)
        points[I, k] = section.points[I] + offsets[k]
        weights[I, k] = section.weights[I] * offset_weights[k]
    end
    points, weights
end

"""
    revolve(section::ControlNet, axis_point::Vec, axis_direction::Vec, angle::Real=2π)

Revolve a 3D curve or surface around an axis. The added parametric direction is
a rational quadratic circular arc.
"""
function revolve(section::ControlNet{3, pdim, T}, axis_point::Vec{3}, axis_direction::Vec{3}, angle::Real=T(2π)) where {pdim, T}
    pdim < 3 || throw(ArgumentError("revolve supports curves and surfaces"))

    iszero(axis_direction) && throw(ArgumentError("axis_direction must be nonzero"))
    0 < abs(angle)/2 ≤ π || throw(ArgumentError("revolve angle must be in (0, 2π]"))
    axis = normalize(axis_direction)
    arc = arcunit(angle)
    points, weights = revolve_points_weights(section, axis_point, axis, arc)

    ControlNet((section.axes..., arc.axes...), points, weights)
end

function revolve_points_weights(section::ControlNet{3, pdim, T}, axis_point::Vec{3}, axis::Vec{3}, arc::ControlNet{2, 1}) where {pdim, T}
    dims = (size(section.points)..., length(arc.points))
    points = Array{Vec{3, T}}(undef, dims)
    weights = Array{T}(undef, dims)
    for I in CartesianIndices(section.points)
        # Split the point into the position on the rotation axis and the
        # perpendicular radius vector.
        x = section.points[I] - axis_point
        h = (x ⋅ axis) * axis # offset
        c = axis_point + h    # center
        r = x - h             # radius
        t = axis × r          # tangent
        for j in eachindex(arc.points)
            p = arc.points[j]
            points[I, j] = c + p[1] * r + p[2] * t
            weights[I, j] = section.weights[I] * arc.weights[j]
        end
    end
    points, weights
end
