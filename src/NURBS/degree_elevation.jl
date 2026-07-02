"""
    elevate(net::ControlNet, degree::Int; direction::Int) -> ControlNet
    elevate(net::ControlNet, degrees::Tuple{Vararg{Int}}) -> ControlNet
    elevate(net::ControlNet; direction::Int, ntimes::Int=1) -> ControlNet

Raise the spline degree in one or more parametric directions without changing
the represented geometry. Rational nets are elevated in homogeneous
coordinates, so both control points and weights are transformed.
"""
function elevate(net::ControlNet{dim, pdim, T}, degree::Int; direction::Int) where {dim, pdim, T}
    check_parametric_direction(direction, pdim)
    axis = net.axes[direction]
    axis_new = elevate(axis, degree)
    axis_new == axis && return net
    points = elevate_values(homogeneous_points(net), axis, axis_new, direction)
    rational_control_net(Base.setindex(net.axes, axis_new, direction), points)
end
function elevate(net::ControlNet{dim, pdim}, degrees::NTuple{pdim, Int}) where {dim, pdim}
    result = net
    for direction in 1:pdim
        result = elevate(result, degrees[direction]; direction)
    end
    result
end
function elevate(net::ControlNet{dim, pdim}; direction::Int, ntimes::Int=1) where {dim, pdim}
    check_parametric_direction(direction, pdim)
    ntimes ≥ 0 || throw(ArgumentError("degree increment must be non-negative"))
    elevate(net, net.axes[direction].degree + ntimes; direction)
end

"""
    elevate(axis::BSplineAxis, degree::Int) -> BSplineAxis

Raise the polynomial degree of a B-spline axis without changing its break
points or continuity.
"""
function elevate(axis::BSplineAxis{T}, degree::Int) where {T}
    p_old = axis.degree
    degree ≥ p_old || throw(ArgumentError("degree must not be lower than the current degree"))
    degree == p_old && return axis

    # Degree elevation keeps the same break points and raises each knot
    # multiplicity by the degree change, preserving the existing continuity.
    multiplicities = knot_multiplicities(axis.knot_vector)
    multiplicities[begin][2] == multiplicities[end][2] == p_old + 1 || throw(ArgumentError("degree elevation requires an open knot vector"))
    for (_, multiplicity) in @view multiplicities[2:end-1]
        multiplicity ≤ p_old || throw(ArgumentError("degree elevation does not support discontinuous interior knots"))
    end

    dp = degree - p_old
    knots = T[]
    sizehint!(knots, length(axis.knot_vector) + dp * length(multiplicities))
    for (value, multiplicity) in multiplicities
        append!(knots, fill(value, multiplicity + dp))
    end
    BSplineAxis(degree, knots)
end

function elevate_values(values::Array{S, N}, axis_old::BSplineAxis{T}, axis_new::BSplineAxis{T}, direction::Int) where {S, N, T}
    axis_new == elevate(axis_old, axis_new.degree) || throw(ArgumentError("new axis must be the elevated old axis"))

    p_old = axis_old.degree
    p_new = axis_new.degree
    dp = p_new - p_old
    order = p_old + 1
    knots_old = axis_old.knot_vector
    knots_new = axis_new.knot_vector
    n_old = nbasis(axis_old)
    n_new = nbasis(axis_new)
    multiplicities = knot_multiplicities(knots_old)
    nspans = length(multiplicities) - 1

    perm = ntuple(Val(N)) do i
        i == 1 && return direction
        ifelse(i ≤ direction, i - 1, i)
    end
    columns_old = reshape(PermutedDimsArray(values, perm), size(values, direction), :)
    columns_new = similar(columns_old, S, n_new, size(columns_old, 2))

    # P[:, d, :] stores the (d-1)-th scaled divided differences of the old
    # control points. Q is the same table for the elevated curve.
    P = Array{S}(undef, n_old, order, size(columns_old, 2))
    Q = Array{S}(undef, n_new, order, size(columns_old, 2))
    fill!(P, zero(S))
    fill!(Q, zero(S))

    for fiber in axes(columns_old, 2)
        for i in 1:n_old
            P[i, 1, fiber] = columns_old[i, fiber]
        end
        for d in 2:order
            l = d - 1
            for i in 1:(n_old-l)
                denominator = knots_old[i+order] - knots_old[i+l]
                denominator > zero(T) || continue
                P[i, d, fiber] = (P[i+1, d-1, fiber] - P[i, d-1, fiber]) / denominator
            end
        end
    end

    # β[span] is the old control index at the left end of each nonzero span,
    # written as a zero-based offset to match the degree-elevation formulas.
    β = Vector{Int}(undef, nspans)
    β[1] = 0
    for span in 2:nspans
        β[span] = β[span-1] + multiplicities[span][2]
    end

    # Scaling between old and elevated divided differences.
    α = Vector{T}(undef, order)
    α[1] = one(T)
    for d in 2:order
        α[d] = α[d-1] * T(order - d + 1) / T(order + dp - d + 1)
    end

    for fiber in axes(columns_old, 2)
        # Boundary divided differences at each span start.
        for span in 1:nspans
            i_old = β[span] + 1
            i_new = β[span] + (span - 1) * dp + 1
            multiplicity = multiplicities[span][2]
            for d in (order-multiplicity+1):order
                Q[i_new, d, fiber] = α[d] * P[i_old, d, fiber]
            end
        end

        # Each elevated span receives `dp` repeated highest-order boundary
        # values before the remaining divided differences are reconstructed.
        for span in 1:nspans
            i_new = β[span] + (span - 1) * dp + 1
            for offset in 1:dp
                Q[i_new+offset, order, fiber] = Q[i_new, order, fiber]
            end
        end

        # Reconstruct lower-order divided differences from right to left. The
        # first layer of Q is the elevated control points.
        for d in order:-1:2
            for i in 1:(n_new-1)
                left = knots_new[i+d-1]
                right = knots_new[i+p_new+1]
                right > left || continue
                Q[i+1, d-1, fiber] = Q[i, d-1, fiber] + (right - left) * Q[i, d, fiber]
            end
        end

        for i in 1:n_new
            columns_new[i, fiber] = Q[i, 1, fiber]
        end
    end

    dims_new = ntuple(i -> i == 1 ? n_new : size(values, perm[i]), Val(N))
    values_new = reshape(columns_new, dims_new)
    permutedims(values_new, invperm(perm))
end
