# Algorithm A2.2: one triangular recurrence evaluates every active B-spline on
# a span. Callers own their dynamic or static workspace and endpoint convention.
@inline function bspline_span_values!(values, left, right, knots, span, p, ξ, lower=nothing)
    @boundscheck checkbounds(knots, (span-p+1):(span+p))
    values[1] = one(eltype(values))
    @inbounds for j in 1:p
        if lower !== nothing && j == p
            for r in 1:p
                lower[r] = values[r]
            end
        end
        left[j] = ξ - knots[span+1-j]
        right[j] = knots[span+j] - ξ
        saved = zero(eltype(values))
        for r in 1:j
            denominator = right[r] + left[j-r+1]
            term = iszero(denominator) ? zero(saved) : values[r] / denominator
            values[r] = saved + right[r] * term
            saved = left[j-r+1] * term
        end
        values[j+1] = saved
    end
    values
end

function greville_abscissae(knots::AbstractVector, p::Int)
    map(1:(length(knots)-p-1)) do i
        sum(j -> knots[i+j], 1:p) / p
    end
end
