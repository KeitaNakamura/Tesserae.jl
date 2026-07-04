"""
    ControlNet(axes::Tuple{Vararg{BSplineAxis}}, points::Array[, weights::Array])

Tensor-product spline control net used while building NURBS geometry.
"""
struct ControlNet{dim, pdim, T, Axes <: NTuple{pdim, BSplineAxis{T}}}
    axes::Axes
    points::Array{Vec{dim, T}, pdim}
    weights::Array{T, pdim}

    function ControlNet{dim, pdim, T, Axes}(axes::Axes, points::Array{Vec{dim, T}, pdim}, weights::Array{T, pdim}) where {dim, pdim, T, Axes <: NTuple{pdim, BSplineAxis{T}}}
        basis_sizes = map(nbasis, axes)
        size(points) == basis_sizes || throw(ArgumentError("points size must match axes"))
        size(weights) == size(points) || throw(ArgumentError("weights size must match points size"))
        new{dim, pdim, T, Axes}(axes, points, weights)
    end
end

function ControlNet(axes::Axes, points::Array{Vec{dim, T}, pdim}, weights::Array{T, pdim}=ones(T, size(points))) where {dim, pdim, T, Axes <: NTuple{pdim, BSplineAxis{T}}}
    ControlNet{dim, pdim, T, Axes}(axes, points, weights)
end

function homogeneous_points(net::ControlNet{dim, pdim, T}) where {dim, pdim, T}
    map(net.points, net.weights) do point, weight
        Vec(ntuple(i -> i ≤ dim ? weight * point[i] : weight, Val(dim + 1)))
    end
end

function rational_control_net(axes::Axes, points::Array{Vec{hdim, T}, pdim}) where {hdim, pdim, T, Axes <: NTuple{pdim, BSplineAxis{T}}}
    weights = map(point -> point[hdim], points)
    control_points = map(points) do point
        weight = point[hdim]
        Vec(ntuple(i -> point[i] / weight, Val(hdim - 1)))
    end
    ControlNet(axes, control_points, weights)
end

"""
    degree(net::ControlNet, direction::Int) -> Int

Return the polynomial degree in one parametric direction.
"""
function degree(net::ControlNet{dim, pdim}, direction::Int) where {dim, pdim}
    check_parametric_direction(direction, pdim)
    degree(net.axes[direction])
end

"""
    knots(net::ControlNet, direction::Int) -> Vector

Return the knot vector in one parametric direction.
"""
function knots(net::ControlNet{dim, pdim}, direction::Int) where {dim, pdim}
    check_parametric_direction(direction, pdim)
    knots(net.axes[direction])
end

"""
    domain(net::ControlNet, direction::Int) -> Tuple

Return the active parametric domain in one parametric direction.
"""
function domain(net::ControlNet{dim, pdim}, direction::Int) where {dim, pdim}
    check_parametric_direction(direction, pdim)
    domain(net.axes[direction])
end

"""
    evaluate(net::ControlNet, ξ::Vec) -> Vec

Evaluate a rational tensor-product B-spline control net at the parametric coordinate `ξ`.
"""
function evaluate(net::ControlNet{dim, pdim, T}, ξ::Vec{pdim}) where {dim, pdim, T}
    supports = map(active_basis, net.axes, Tuple(ξ))
    indices = map(first, supports)
    values = map(last, supports)

    p = zero(Vec{dim, T})
    w = zero(T)
    for I in CartesianIndices(map(length, values))
        ids = map(getindex, indices, Tuple(I))
        basis_value = prod(map(getindex, values, Tuple(I)))
        weight = basis_value * net.weights[ids...]
        p += weight * net.points[ids...]
        w += weight
    end

    p / w
end

"""
    reverse(net::ControlNet; direction::Int=1) -> ControlNet

Reverse the parametrization in one direction without changing the geometry.
"""
function Base.reverse(net::ControlNet{dim, pdim, T}; direction::Int=1) where {dim, pdim, T}
    check_parametric_direction(direction, pdim)
    ControlNet(
        Base.setindex(net.axes, reverse(net.axes[direction]), direction),
        reverse(net.points; dims=direction),
        reverse(net.weights; dims=direction),
    )
end

"""
    boundaries(net::ControlNet) -> Tuple

Extract all boundary control nets.
"""
function boundaries(net::ControlNet{dim, 2}) where {dim}
    tuple(
        boundaries(net, 1, -1),
        boundaries(net, 1, +1),
        boundaries(net, 2, -1),
        boundaries(net, 2, +1),
    )
end

function boundaries(net::ControlNet{dim, 3}) where {dim}
    tuple(
        boundaries(net, 1, -1),
        boundaries(net, 1, +1),
        boundaries(net, 2, -1),
        boundaries(net, 2, +1),
        boundaries(net, 3, -1),
        boundaries(net, 3, +1),
    )
end

"""
    boundaries(net::ControlNet, direction::Int, side::Int) -> ControlNet

Extract one boundary control net. `direction` is the fixed parametric
direction; `side` is -1 for the lower end and +1 for the upper end.
"""
function boundaries(net::ControlNet{dim, pdim}, direction::Int, side::Int) where {dim, pdim}
    2 ≤ pdim ≤ 3 || throw(ArgumentError("boundaries supports surfaces and volumes"))
    1 ≤ direction ≤ pdim || throw(ArgumentError("direction must be between 1 and the parametric dimension"))
    (side == -1 || side == 1) || throw(ArgumentError("side must be -1 or +1"))

    fixed = side == -1 ? firstindex(net.points, direction) : lastindex(net.points, direction)

    control_net_slice(net, dropat(net.axes, direction), direction, fixed)
end

# Build a ControlNet from one directional slice of local tensor-product arrays.
function control_net_slice(net::ControlNet{dim, pdim, T}, axes::SliceAxes, direction::Int, selection) where {dim, pdim, T, slice_pdim, SliceAxes <: NTuple{slice_pdim, BSplineAxis{T}}}
    ControlNet(
        axes,
        copy(selectdim(net.points, direction, selection)),
        copy(selectdim(net.weights, direction, selection)),
    )
end
