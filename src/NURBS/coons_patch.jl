"""
    coons_patch(bottom, top, left, right) -> ControlNet

Build a tensor-product surface from four boundary curves. `bottom` and `top`
run in the first parametric direction; `left` and `right` run in the second.
"""
function coons_patch(bottom::ControlNet{dim, 1}, top::ControlNet{dim, 1}, left::ControlNet{dim, 1}, right::ControlNet{dim, 1}) where {dim}
    check_coons_corners(bottom, top, left, right)

    axis_u = merge(bottom.axes[1], top.axes[1])
    axis_v = merge(left.axes[1], right.axes[1])
    axes = (axis_u, axis_v)

    bottom = match_basis(bottom, (axis_u,))
    top = match_basis(top, (axis_u,))
    left = match_basis(left, (axis_v,))
    right = match_basis(right, (axis_v,))

    bottom_top = match_basis(loft([bottom, top]), axes)
    left_right = match_basis(swap_directions(loft([left, right])), axes)
    corners = match_basis(coons_corner(bottom, top), axes)

    coons_blend(axes, bottom_top, left_right, corners)
end

function coons_blend(axes, bottom_top::ControlNet, left_right::ControlNet, corners::ControlNet)
    weights = @. bottom_top.weights + left_right.weights - corners.weights
    weighted_points = @. bottom_top.weights * bottom_top.points
    @. weighted_points += left_right.weights * left_right.points
    @. weighted_points -= corners.weights * corners.points

    all(!iszero, weights) || throw(ArgumentError("combined rational weights must be nonzero"))
    ControlNet(axes, weighted_points ./ weights, weights)
end

function match_basis(net::ControlNet{dim, pdim}, axes::NTuple{pdim, BSplineAxis}) where {dim, pdim}
    result = net
    for direction in 1:pdim
        axis = axes[direction]
        # Coons blending only needs compatible superspaces: degree elevation
        # followed by knot insertion.
        result = elevate(result, axis.degree; direction)
        result.axes[direction] == axis && continue
        result = refineto(result, axis; direction)
    end
    result
end

function swap_directions(net::ControlNet{dim, 2}) where {dim}
    ControlNet((net.axes[2], net.axes[1]), permutedims(net.points), permutedims(net.weights))
end

function coons_corner(bottom::ControlNet{dim, 1, T}, top::ControlNet{dim, 1, T}) where {dim, T}
    axis_u = open_bspline_axis(T, linear, 1)
    axis_v = open_bspline_axis(T, linear, 1)
    ControlNet(
        (axis_u, axis_v),
        reshape([bottom.points[begin], bottom.points[end], top.points[begin], top.points[end]], 2, 2),
        reshape([bottom.weights[begin], bottom.weights[end], top.weights[begin], top.weights[end]], 2, 2),
    )
end

function check_coons_corners(bottom::ControlNet, top::ControlNet, left::ControlNet, right::ControlNet)
    check_matching_corner(bottom, firstindex(bottom.points), left, firstindex(left.points), "bottom-left")
    check_matching_corner(bottom, lastindex(bottom.points), right, firstindex(right.points), "bottom-right")
    check_matching_corner(top, firstindex(top.points), left, lastindex(left.points), "top-left")
    check_matching_corner(top, lastindex(top.points), right, lastindex(right.points), "top-right")
    nothing
end

function check_matching_corner(a::ControlNet, ia::Integer, b::ControlNet, ib::Integer, name::String)
    isapprox(a.points[ia], b.points[ib])   || throw(ArgumentError("Coons patch $name corner points must match"))
    isapprox(a.weights[ia], b.weights[ib]) || throw(ArgumentError("Coons patch $name corner weights must match"))
    nothing
end
