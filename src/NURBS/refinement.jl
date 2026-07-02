"""
    insert_knot(net::ControlNet, ξ; direction, ntimes=1) -> ControlNet

Insert `ξ` one or more times in a parametric direction without changing the
represented geometry.
"""
function insert_knot(net::ControlNet{dim, pdim, T}, ξ::T; direction::Int, ntimes::Int=1) where {dim, pdim, T}
    check_parametric_direction(direction, pdim)
    ntimes ≥ 0 || throw(ArgumentError("number of knot insertions must be non-negative"))

    axis_new = net.axes[direction]
    points = homogeneous_points(net)
    for _ in 1:ntimes
        axis = axis_new
        axis_new = insert_knot(axis, ξ)
        points = insert_knot_values(points, axis, ξ, direction)
    end
    rational_control_net(Base.setindex(net.axes, axis_new, direction), points)
end

"""
    insert_knot(net::ControlNet, knots::AbstractVector; direction) -> ControlNet

Insert each knot in `knots` in a parametric direction without changing the
represented geometry. Repeated entries request repeated insertions.
"""
function insert_knot(net::ControlNet{dim, pdim, T}, knots::AbstractVector{T}; direction::Int) where {dim, pdim, T}
    check_parametric_direction(direction, pdim)

    axis_new = net.axes[direction]
    points = homogeneous_points(net)
    for ξ in knots
        axis = axis_new
        axis_new = insert_knot(axis, ξ)
        points = insert_knot_values(points, axis, ξ, direction)
    end
    rational_control_net(Base.setindex(net.axes, axis_new, direction), points)
end

# Refine one parametric direction until its axis matches the requested axis.
function refineto(net::ControlNet{dim, pdim, T}, axis_new::BSplineAxis{T}; direction::Int) where {dim, pdim, T}
    check_parametric_direction(direction, pdim)
    axis_old = net.axes[direction]
    axis_old.degree == axis_new.degree || throw(ArgumentError("knot insertion requires matching degrees"))

    knots = T[]
    # Refine the selected direction until its knot vector matches the requested
    # axis. The degree must already match; this is only knot insertion.
    for (ξ, multiplicity) in knot_multiplicities(axis_new.knot_vector)
        ninsertions = multiplicity - count(==(ξ), axis_old.knot_vector)
        ninsertions ≥ 0 || throw(ArgumentError("new knot vector must contain the old knot vector"))
        append!(knots, fill(ξ, ninsertions))
    end
    insert_knot(axis_old, knots) == axis_new || throw(ArgumentError("new axis must be produced by knot insertions from old axis"))

    points = refine_values(homogeneous_points(net), axis_old, axis_new, knots, direction)
    rational_control_net(Base.setindex(net.axes, axis_new, direction), points)
end

function refine_values(values::Array{S, N}, axis_old::BSplineAxis{T}, axis_new::BSplineAxis{T}, knots::Vector{T}, direction::Int) where {S, N, T}
    p = axis_old.degree
    knot_vector_old = axis_old.knot_vector
    knot_vector_new = axis_new.knot_vector
    n_old = nbasis(axis_old)
    n_new = nbasis(axis_new)

    perm = ntuple(Val(N)) do i
        i == 1 && return direction
        ifelse(i ≤ direction, i - 1, i)
    end
    columns_old = reshape(PermutedDimsArray(values, perm), size(values, direction), :)
    columns_new = similar(columns_old, S, n_new, size(columns_old, 2))

    isempty(knots) && return values

    # Standard knot-refinement pass. The inserted knots are handled in one
    # backward sweep, so each tensor-product fiber is copied only once.
    r = length(knots) - 1

    # Only the basis functions between the first and last inserted knots change.
    # The untouched leading/trailing controls are copied before the sweep.
    a = searchsortedlast(knot_vector_old, first(knots)) - 1
    b = searchsortedlast(knot_vector_old, last(knots))

    for fiber in axes(columns_old, 2)
        # Controls outside the refined knot interval are unchanged.
        for j in 0:(a-p)
            columns_new[j+1, fiber] = columns_old[j+1, fiber]
        end
        for j in (b-1):(n_old-1)
            columns_new[j+length(knots)+1, fiber] = columns_old[j+1, fiber]
        end

        # `i` walks the old controls from right to left; `k` walks the new
        # controls, leaving room for the inserted knots as they are processed.
        i = b + p - 1
        k = b + p + r
        for j in (r+1):-1:1
            ξ = knots[j]

            # Move old controls that lie to the right of the current inserted
            # knot into their final positions.
            while i > a && ξ ≤ knot_vector_old[i+1]
                columns_new[k-p, fiber] = columns_old[i-p, fiber]
                k -= 1
                i -= 1
            end

            # Update the p affected controls by the usual knot-insertion
            # convex combinations.
            columns_new[k-p, fiber] = columns_new[k-p+1, fiber]
            for l in 1:p
                row = k - p + l
                numerator = knot_vector_new[k+l+1] - ξ
                if iszero(numerator)
                    columns_new[row, fiber] = columns_new[row+1, fiber]
                else
                    α = numerator / (knot_vector_new[k+l+1] - knot_vector_old[i-p+l+1])
                    columns_new[row, fiber] = α * columns_new[row, fiber] + (1 - α) * columns_new[row+1, fiber]
                end
            end
            k -= 1
        end
    end

    dims_new = ntuple(i -> i == 1 ? n_new : size(values, perm[i]), Val(N))
    values_new = reshape(columns_new, dims_new)
    permutedims(values_new, invperm(perm))
end

function insert_knot_values(values::Array{S, N}, axis::BSplineAxis{T}, ξ::T, direction::Int) where {S, N, T}
    knot_vector = axis.knot_vector
    p = axis.degree
    multiplicity = count(==(ξ), knot_vector)
    span = searchsortedlast(knot_vector, ξ)

    perm = ntuple(Val(N)) do i
        i == 1 && return direction
        ifelse(i ≤ direction, i - 1, i)
    end
    columns_old = reshape(PermutedDimsArray(values, perm), size(values, direction), :)
    columns_new = similar(columns_old, S, size(columns_old, 1) + 1, size(columns_old, 2))

    for col in axes(columns_old, 2)
        for row in 1:(span-p)
            columns_new[row, col] = columns_old[row, col]
        end
        for row in (span-multiplicity+1):size(columns_new, 1)
            columns_new[row, col] = columns_old[row-1, col]
        end
        for row in (span-p+1):(span-multiplicity)
            α = (ξ - knot_vector[row]) / (knot_vector[row+p] - knot_vector[row])
            columns_new[row, col] = α * columns_old[row, col] + (1 - α) * columns_old[row-1, col]
        end
    end

    dims_new = ntuple(i -> i == 1 ? size(columns_new, 1) : size(values, perm[i]), Val(N))
    values_new = reshape(columns_new, dims_new)
    permutedims(values_new, invperm(perm))
end

"""
    insert_knot(axis::BSplineAxis, ξ) -> BSplineAxis

Return a B-spline axis with `ξ` inserted once into the knot vector.
"""
function insert_knot(axis::BSplineAxis{T}, ξ::T) where {T}
    check_insertable_knot(axis, ξ)
    knot_vector = axis.knot_vector
    p = axis.degree
    multiplicity = count(==(ξ), knot_vector)
    multiplicity < p || throw(ArgumentError("knot multiplicity cannot exceed the degree"))

    # `span` is the last position whose knot value is ≤ ξ. Inserting after it
    # preserves sorted order and makes the control-point update use standard
    # one-based B-spline indexing.
    span = searchsortedlast(knot_vector, ξ)
    knot_vector_new = Vector{T}(undef, length(knot_vector) + 1)
    for i in 1:span
        knot_vector_new[i] = knot_vector[i]
    end
    knot_vector_new[span+1] = ξ
    for i in (span+1):length(knot_vector)
        knot_vector_new[i+1] = knot_vector[i]
    end

    BSplineAxis(axis.degree, knot_vector_new)
end

"""
    insert_knot(axis::BSplineAxis, knots::AbstractVector) -> BSplineAxis

Return a B-spline axis with each knot in `knots` inserted in order.
"""
function insert_knot(axis::BSplineAxis{T}, knots::AbstractVector{T}) where {T}
    result = axis
    for ξ in knots
        result = insert_knot(result, ξ)
    end
    result
end

function check_insertable_knot(axis::BSplineAxis, ξ::Real)
    knot_vector = axis.knot_vector
    p = axis.degree
    knot_vector[p+1] < ξ < knot_vector[end-p] ||
        throw(ArgumentError("inserted knot must be strictly inside the parametric domain"))
    nothing
end

"""
    refine(net::ControlNet, ninsertions::Tuple{Vararg{Integer}}) -> ControlNet

Uniformly refine each parametric direction. Each entry gives the number of
knots inserted into every nonzero span in that direction.
"""
function refine(net::ControlNet{dim, pdim}, ninsertions::NTuple{pdim, <: Integer}) where {dim, pdim}
    result = net
    for direction in 1:pdim
        n = ninsertions[direction]
        axis = result.axes[direction]
        axis_new = refine(axis, n)
        axis_new == axis && continue
        result = refineto(result, axis_new; direction)
    end
    result
end

"""
    refine(net::ControlNet, n::Integer; direction::Int) -> ControlNet

Uniformly refine one parametric direction by inserting `n` knots into every nonzero span.
"""
function refine(net::ControlNet{dim, pdim}, n::Integer; direction::Int) where {dim, pdim}
    check_parametric_direction(direction, pdim)
    refine(net, ntuple(d -> d == direction ? n : 0, Val(pdim)))
end

"""
    refine(axis::BSplineAxis, n::Integer) -> BSplineAxis

Uniformly refine each nonzero knot span of `axis`.
"""
function refine(axis::BSplineAxis{T}, n::Integer) where {T}
    n ≥ 0 || throw(ArgumentError("number of knot insertions per span must be non-negative"))
    n == 0 && return axis

    p = axis.degree
    knot_vector = axis.knot_vector
    knots = T[]

    # Positive knot spans become elements. Insert the internal subdivision
    # points of each such span, leaving repeated knots and boundaries intact.
    for span in (p+1):(length(knot_vector)-p-1)
        left = knot_vector[span]
        right = knot_vector[span+1]
        left < right || continue

        for i in 1:n
            push!(knots, left + (right - left) * T(i) / T(n + 1))
        end
    end

    insert_knot(axis, knots)
end

"""
    split(net::ControlNet, direction::Int, ξ) -> Tuple{ControlNet, ControlNet}

Split a control net at an interior parametric coordinate. The knot is inserted
until its multiplicity reaches the degree, then the cut becomes an open end of
both returned nets.
"""
function Base.split(net::ControlNet{dim, pdim, T}, direction::Int, ξ::T) where {dim, pdim, T}
    check_parametric_direction(direction, pdim)

    axis = net.axes[direction]
    refined = insert_knot(net, ξ; direction, ntimes=axis.degree - count(==(ξ), axis.knot_vector))
    axis_l, axis_r = split(refined.axes[direction], ξ)
    point_axis = axes(refined.points, direction)
    range_l = first(point_axis):(first(point_axis) + nbasis(axis_l) - 1)
    range_r = (last(point_axis) - nbasis(axis_r) + 1):last(point_axis)

    tuple(
        control_net_slice(refined, Base.setindex(refined.axes, axis_l, direction), direction, range_l),
        control_net_slice(refined, Base.setindex(refined.axes, axis_r, direction), direction, range_r),
    )
end

"""
    split(axis::BSplineAxis, ξ) -> Tuple{BSplineAxis, BSplineAxis}

Split a B-spline axis at an interior knot.
"""
function Base.split(axis::BSplineAxis{T}, ξ::T) where {T}
    check_insertable_knot(axis, ξ)
    knot_vector = axis.knot_vector
    first_cut = searchsortedfirst(knot_vector, ξ)
    last_cut = searchsortedlast(knot_vector, ξ)
    knot = knot_vector[first_cut]

    knot_vector_l = vcat(knot_vector[begin:last_cut], knot)
    knot_vector_r = vcat(knot, knot_vector[first_cut:end])
    BSplineAxis(axis.degree, knot_vector_l), BSplineAxis(axis.degree, knot_vector_r)
end
