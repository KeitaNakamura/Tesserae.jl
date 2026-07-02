"""
    BSplineAxis(degree::Int, knot_vector::Vector) -> BSplineAxis

One parametric direction of a B-spline basis, defined by a polynomial `degree`
and a `knot_vector`.
"""
struct BSplineAxis{T}
    degree::Int
    knot_vector::Vector{T}

    function BSplineAxis{T}(degree::Int, knot_vector::Vector{T}) where {T}
        degree < 0 && throw(ArgumentError("degree must be non-negative"))
        length(knot_vector) ≥ degree + 2 || throw(ArgumentError("knot vector length must be at least degree + 2"))
        issorted(knot_vector) || throw(ArgumentError("knot vector must be sorted"))
        new{T}(degree, knot_vector)
    end
end

function BSplineAxis(degree::Int, knot_vector::Vector{T}) where {T}
    BSplineAxis{T}(degree, knot_vector)
end

"""
    degree(axis::BSplineAxis) -> Int

Return the polynomial degree of a B-spline axis.
"""
degree(axis::BSplineAxis) = axis.degree

"""
    knots(axis::BSplineAxis) -> Vector

Return the knot vector of a B-spline axis.
"""
knots(axis::BSplineAxis) = axis.knot_vector

"""
    domain(axis::BSplineAxis) -> Tuple

Return the active parametric domain of a B-spline axis.
"""
function domain(axis::BSplineAxis)
    p = degree(axis)
    knot_vector = knots(axis)
    knot_vector[p+1], knot_vector[end-p]
end

function Base.:(==)(left::BSplineAxis, right::BSplineAxis)
    left.degree == right.degree && left.knot_vector == right.knot_vector
end

"""
    reverse(axis::BSplineAxis) -> BSplineAxis

Reverse the parametric direction of a B-spline axis.
"""
function Base.reverse(axis::BSplineAxis)
    knot_vector = axis.knot_vector
    lower = first(knot_vector)
    upper = last(knot_vector)
    BSplineAxis(axis.degree, map(knot -> lower + upper - knot, reverse(knot_vector)))
end

"""
    merge(a::BSplineAxis, b::BSplineAxis) -> BSplineAxis

Return a common B-spline axis containing both input spline spaces.
"""
function Base.merge(a::BSplineAxis, b::BSplineAxis)
    degree = max(a.degree, b.degree)
    a = elevate(a, degree)
    b = elevate(b, degree)
    BSplineAxis(degree, merge_knot_vectors(a.knot_vector, b.knot_vector))
end

# Take the union of knot values and keep the larger multiplicity, giving the
# common knot vector both axes can be refined to.
function merge_knot_vectors(left::Vector{T}, right::Vector{T}) where {T}
    knots = T[]
    for knot in sort(unique(vcat(left, right)))
        multiplicity = max(count(==(knot), left), count(==(knot), right))
        append!(knots, fill(knot, multiplicity))
    end
    knots
end

"""
    nbasis(axis::BSplineAxis) -> Int

Return the number of one-dimensional B-spline basis functions defined by `axis`.
"""
nbasis(axis::BSplineAxis) = length(axis.knot_vector) - axis.degree - 1

"""
    open_bspline_axis(degree::Int, nspans::Integer) -> BSplineAxis
    open_bspline_axis(::Type{T}, degree::Int, nspans::Integer) where {T} -> BSplineAxis

Create a B-spline axis with an open uniform knot vector, polynomial `degree`,
and `nspans` nonzero knot intervals.
"""
open_bspline_axis(degree::Int, nspans::Integer) = open_bspline_axis(Float64, degree, nspans)
function open_bspline_axis(::Type{T}, degree::Int, nspans::Integer) where {T}
    degree < 1 && throw(ArgumentError("degree must be positive"))
    nspans ≥ 1 || throw(ArgumentError("nspans must be positive"))

    # Open knot vectors interpolate the first and last control points. Interior
    # knots are uniform for this primitive generator.
    knot_vector = vcat(zeros(T, degree+1), T[i/nspans for i in 1:(nspans-1)], ones(T, degree+1))
    BSplineAxis(degree, knot_vector)
end

"""
    greville_abscissae(axis::BSplineAxis) -> Vector

Return the Greville abscissae associated with the B-spline basis functions
defined by `axis`.
"""
function greville_abscissae(axis::BSplineAxis)
    p = axis.degree
    # Greville abscissae are the standard parametric locations associated with
    # control points of a B-spline curve.
    map(1:nbasis(axis)) do i
        sum(j -> axis.knot_vector[i+j], 1:p) / p
    end
end

"""
    knot_multiplicities(knot_vector::Vector) -> Vector{Tuple}

Return each distinct knot value in `knot_vector` with its multiplicity.
"""
function knot_multiplicities(knot_vector::Vector{T}) where {T}
    multiplicities = Tuple{T, Int}[]
    i = firstindex(knot_vector)
    while i ≤ lastindex(knot_vector)
        value = knot_vector[i]
        multiplicity = 1
        while i + multiplicity ≤ lastindex(knot_vector) && knot_vector[i + multiplicity] == value
            multiplicity += 1
        end

        push!(multiplicities, (value, multiplicity))
        i += multiplicity
    end
    multiplicities
end

"""
    knot_span(axis::BSplineAxis, ξ::Real) -> Int

Return the knot span containing the parametric coordinate `ξ`.
"""
function knot_span(axis::BSplineAxis{T}, ξ::Real) where {T}
    p = axis.degree
    u = T(ξ)
    lower = axis.knot_vector[p+1]
    upper = axis.knot_vector[end-p]
    lower ≤ u ≤ upper || throw(ArgumentError("knot coordinate must be inside the parametric domain"))
    clamp(searchsortedlast(axis.knot_vector, u), p+1, nbasis(axis))
end

"""
    active_basis(axis::BSplineAxis, ξ::Real) -> Tuple{UnitRange{Int}, Vector}

Return the active basis indices and their values at the parametric coordinate
`ξ`.

For polynomial degree `p`, only `p + 1` B-spline basis functions are nonzero at
`ξ`. The returned `indices` identify those basis functions, and `values` stores
the corresponding basis values in the same order.
"""
function active_basis(axis::BSplineAxis{T}, ξ::Real) where {T}
    p = axis.degree
    span = knot_span(axis, ξ)
    indices = (span-p):span
    values = zeros(T, p+1)
    values[1] = one(T)
    p == 0 && return indices, values

    u = T(ξ)
    left = Vector{T}(undef, p)
    right = Vector{T}(undef, p)
    # Algorithm A2.2 from The NURBS Book: evaluate the nonzero basis functions
    # on one span using a stable triangular recurrence.
    for j in 1:p
        left[j] = u - axis.knot_vector[span+1-j]
        right[j] = axis.knot_vector[span+j] - u
        saved = zero(T)
        for r in 1:j
            denominator = right[r] + left[j-r+1]
            term = iszero(denominator) ? zero(T) : values[r] / denominator
            values[r] = saved + right[r] * term
            saved = left[j-r+1] * term
        end
        values[j+1] = saved
    end

    indices, values
end

"""
    basis_matrix(axis::BSplineAxis, points::AbstractVector) -> Matrix

Evaluate all B-spline basis functions defined by `axis` at each parametric
coordinate in `points`.

The returned matrix has one row per point and one column per basis function.
Entry `(i, j)` is the value of the `j`-th basis function at `points[i]`.
"""
function basis_matrix(axis::BSplineAxis{T}, points::AbstractVector) where {T}
    values = zeros(T, length(points), nbasis(axis))
    for (row, ξ) in pairs(points)
        # Only p + 1 basis functions are nonzero at a point, so fill the dense
        # matrix row from the active span values.
        indices, local_values = active_basis(axis, ξ)
        for (id, value) in zip(indices, local_values)
            values[row, id] = value
        end
    end
    values
end
