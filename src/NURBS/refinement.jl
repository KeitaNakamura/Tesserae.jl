"""
    insert_knot(net::ControlNet, ξ; direction, ntimes=1) -> ControlNet

Insert `ξ` one or more times in a parametric direction without changing the
represented geometry.
"""
function insert_knot(net::ControlNet{dim, pdim, T}, ξ::T; direction::Int, ntimes::Int=1) where {dim, pdim, T}
    ntimes ≥ 0 || throw(ArgumentError("number of knot insertions must be non-negative"))
    insert_knot(net, fill(ξ, ntimes); direction)
end

"""
    insert_knot(net::ControlNet, knots::AbstractVector; direction) -> ControlNet

Insert each knot in `knots` in a parametric direction without changing the
represented geometry. Repeated entries request repeated insertions.
"""
function insert_knot(net::ControlNet{dim, pdim, T}, knots::AbstractVector{T}; direction::Int) where {dim, pdim, T}
    check_parametric_direction(direction, pdim)

    refine_control_net(net, knots, direction)
end

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

    refine_control_net(net, knots, direction, axis_new)
end

function refine_control_net(net::ControlNet, knots, direction::Int, axis_new=nothing)
    axis_old = net.axes[direction]
    points = homogeneous_points(net)
    if length(knots) <= 1
        axis_new = axis_new === nothing ? insert_knot(axis_old, knots) : axis_new
        isempty(knots) || (points = insert_knot_values(points, axis_old, only(knots), direction))
    elseif issorted(knots) && can_batch_insert(points, axis_old)
        axis_new = axis_new === nothing ? insert_knot(axis_old, knots) : axis_new
        points = refine_values(points, axis_old, axis_new, knots, direction)
    else
        # Preserve the requested order and the original arithmetic for signed,
        # zero, nonfinite or subnormal weights and generic scalar arithmetic.
        # Cancellation can otherwise turn a small weight into zero, introducing
        # an infinite control point when homogeneous coordinates are divided.
        for ξ in knots
            axis_next = insert_knot(axis_old, ξ)
            points = insert_knot_values(points, axis_old, ξ, direction)
            axis_old = axis_next
        end
        axis_new = axis_new === nothing ? axis_old : axis_new
    end
    rational_control_net(Base.setindex(net.axes, axis_new, direction), points)
end

function can_batch_insert(points::AbstractArray{<:Vec{dim,T}}, axis::BSplineAxis) where {dim,T}
    T <: AbstractFloat && isbitstype(T) || return false
    # The forward and backward formulas need not propagate overflowing knot
    # differences identically, even when every knot itself is finite.
    isfinite(last(axis.knot_vector) - first(axis.knot_vector)) || return false
    minimum_weight = floatmin(T)
    all(point -> all(isfinite, point) && point[dim] ≥ minimum_weight, points)
end

@inline function refine_values(values::Array{S,N}, axis_old::BSplineAxis{T}, axis_new::BSplineAxis{T}, knots::AbstractVector{T}, direction::Int) where {S,N,T}
    isempty(knots) && return values
    # Reuse coefficients across many fibers, while keeping the cache small
    # relative to the control net. Short curves need no cache allocation.
    nfibers = N == 1 ? 1 : length(values) ÷ size(values, direction)
    coefficients = nfibers > max(16, 4 * axis_old.degree) ?
        Matrix{Tuple{T,T}}(undef, axis_old.degree, length(knots)) : nothing
    refine_values(values, axis_old, axis_new, knots, direction, coefficients)
end

function refine_values(values, axis_old, axis_new, knots, direction, coefficients)
    a = searchsortedlast(axis_old.knot_vector, first(knots)) - 1
    b = searchsortedlast(axis_old.knot_vector, last(knots))
    map_fibers(values, direction, nbasis(axis_new)) do columns_new, columns_old
        if size(columns_old, 2) > 0
            # The first fiber fills an optional cache; all others reuse it.
            refine_fiber!(columns_new, columns_old, axis_old, axis_new, knots, a, b, 1, coefficients, Val(coefficients !== nothing))
            for fiber in 2:size(columns_old, 2)
                refine_fiber!(columns_new, columns_old, axis_old, axis_new, knots, a, b, fiber, coefficients, Val(false))
            end
        end
    end
end

@inline function refine_fiber!(columns_new, columns_old, axis_old, axis_new, knots, a, b, fiber, coefficients, cache_mode)
    p = axis_old.degree
    knot_vector_old, knot_vector_new = axis_old.knot_vector, axis_new.knot_vector
    n_old, r = nbasis(axis_old), length(knots) - 1
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
        ξ = knots[firstindex(knots) + j - 1]

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
            right = knot_vector_new[k+l+1]
            if right == ξ
                columns_new[row, fiber] = columns_new[row+1, fiber]
            else
                # The same coefficient pair is used for single insertions.
                α, β = cached_insertion_coefficients!(coefficients, cache_mode, l, j, knot_vector_old[i-p+l+1], ξ, right)
                columns_new[row, fiber] = β * columns_new[row, fiber] + α * columns_new[row+1, fiber]
            end
        end
        k -= 1
    end
end

@inline cached_insertion_coefficients!(::Nothing, ::Val{false}, l, j, left, ξ, right) = insertion_coefficients(left, ξ, right)
@inline function cached_insertion_coefficients!(coefficients::Matrix, ::Val{true}, l, j, left, ξ, right)
    @inbounds coefficients[l,j] = insertion_coefficients(left, ξ, right)
end
@inline function cached_insertion_coefficients!(coefficients::Matrix, ::Val{false}, l, j, left, ξ, right)
    @inbounds coefficients[l,j]
end

@inline function insertion_coefficients(left, ξ, right)
    # Retain generic scalar arithmetic: forming the right gap unnecessarily
    # can overflow a fixed-size Rational even when both coefficients fit.
    denominator = right - left
    α = (ξ - left) / denominator
    β = 1 - α
    if iszero(β)
        β = (right - ξ) / denominator
        α = 1 - β
    end
    α, β
end

@inline function insertion_coefficients(left::AbstractFloat, ξ::AbstractFloat, right::AbstractFloat)
    denominator = right - left
    left_gap = ξ - left
    right_gap = right - ξ
    # Always compute the smaller coefficient directly. A fixed near-zero
    # threshold still loses relative accuracy just outside that threshold.
    # Overflowing intervals retain the sequential path's forward arithmetic.
    left_is_small = (left_gap <= right_gap) | !isfinite(denominator)
    small = ifelse(left_is_small, left_gap, right_gap) / denominator
    large = 1 - small
    ifelse(left_is_small, small, large), ifelse(left_is_small, large, small)
end

function insert_knot_values(values::Array{S, N}, axis::BSplineAxis{T}, ξ::T, direction::Int) where {S, N, T}
    knot_vector = axis.knot_vector
    p = axis.degree
    multiplicity = count(==(ξ), knot_vector)
    span = searchsortedlast(knot_vector, ξ)

    map_fibers(values, direction, size(values, direction) + 1) do columns_new, columns_old
        for col in axes(columns_old, 2)
            for row in 1:(span-p)
                columns_new[row, col] = columns_old[row, col]
            end
            for row in (span-multiplicity+1):size(columns_new, 1)
                columns_new[row, col] = columns_old[row-1, col]
            end
            for row in (span-p+1):(span-multiplicity)
                α, β = insertion_coefficients(knot_vector[row], ξ, knot_vector[row+p])
                columns_new[row, col] = α * columns_old[row, col] + β * columns_old[row-1, col]
            end
        end
    end
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
