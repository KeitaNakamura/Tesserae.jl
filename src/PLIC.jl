module PLIC

using ..Tesserae: Vec

# Analytic plane/box geometry for PLIC, following Scardovelli and Zaleski,
# J. Comput. Phys. 164 (2000), 228-237, doi:10.1006/jcph.2000.6567.
# Degenerate orientations are covered by the lower-dimensional branches;
# see also Lehmann and Gekle, Computation 10 (2022), 21, Section 2.

# The geometric kernels share one normal-domain check and scaling factor.
function normal_scale(n::Vec{N, T}) where {N, T<:AbstractFloat}
    N in (2, 3) || throw(ArgumentError("plane/box geometry supports two or three dimensions"))
    all(isfinite, n) || throw(ArgumentError("the plane normal must be finite"))
    scale = maximum(abs, n)
    scale > 0 || throw(ArgumentError("the plane normal must be nonzero"))
    scale
end

# Reflect, scale, and sort the normal: 0 ≤ a ≤ b ≤ c = 1.
function plic_axes(n::Vec{N, T}) where {N, T<:AbstractFloat}
    scale = normal_scale(n)
    a, b, c = N == 2 ? (abs(n[1])/scale, abs(n[2])/scale, zero(T)) : abs.(n)./scale
    a, b = minmax(a, b)
    b, c = minmax(b, c)
    a, b = minmax(a, b)
    a, b, c, scale
end

# Volume measured from the nearest corner; 0 ≤ t ≤ (a+b+c)/2.
function plic_corner_fraction(t, a, b, c)
    t ≤ 0 && return zero(t)
    if a+b ≤ t ≤ c
        return (t-(a+b)/2)/c
    elseif t < a
        return (t/a)*(t/b)*(t/c)/6
    end
    # Factor t^3-(t-a)^3 before division by a. This avoids cancellation
    # and also gives the exact two-dimensional limit when a is zero.
    value = 3*(t/b)*(t-a)+(a/b)*a
    if t > b
        r, s = t-b, max(t-c, zero(t))
        value -= (r/a)*(r/b)*r+(s/a)*(s/b)*s
    end
    value/(6c)
end

"""
    plane_fraction(n, α)

Fraction of the centered unit square or cube `[-1/2,1/2]^N` satisfying
`n⋅x ≤ α`. In 2D the boundary is a line and the fraction is an area fraction;
in 3D it is a plane and a volume fraction. `n` is a nonzero, finite two- or
three-dimensional `Vec`; its length need not be normalized. The cut is
computed analytically in the input precision.
"""
function plane_fraction(n::Vec{N, T}, α::T) where {N, T<:AbstractFloat}
    isnan(α) && throw(ArgumentError("the plane offset must not be NaN"))
    a, b, c, scale = plic_axes(n)
    halfspan = (a+b+c)/2
    offset = α/scale
    offset ≤ -halfspan && return zero(T)
    offset ≥ halfspan && return one(T)
    iszero(offset) && return one(T)/2
    fraction = plic_corner_fraction(halfspan-abs(offset), a, b, c)
    offset < 0 ? fraction : one(T)-fraction
end

"""
    plane_offset(n, fraction)

Find `α` such that `plane_fraction(n, α) == fraction`, up to roundoff.
`fraction` must lie in `[0,1]`. Empty and full cells return the supporting
lines or planes of the square or cube; a half-full cell returns zero.
All branches are analytic.
"""
function plane_offset(n::Vec{N, T}, fraction::T) where {N, T<:AbstractFloat}
    0 ≤ fraction ≤ 1 || throw(ArgumentError("the volume fraction must lie in [0,1]"))
    if iszero(fraction) || fraction == 1
        scale = normal_scale(n)
        span = scale*(sum(abs, n./scale)/2)
        return iszero(fraction) ? -span : span
    end
    a, b, c, scale = plic_axes(n)
    halfspan = (a+b+c)/2
    fraction == one(T)/2 && return zero(T)
    v = min(fraction, one(T)-fraction)
    if a+b ≤ 2c*v
        # Linear slab: four cube corners lie inside the plane.
        offset = c*(v-one(T)/2)
    elseif 6b*c*v < a*a
        # Tetrahedron entirely contained in the cube.
        offset = (6a*b*c*v)^(one(T)/3)-halfspan
    elseif v ≤ (3*(b-a)+(a/b)*a)/(6c)
        # Tetrahedron with one corner clipped by the cube.
        offset = a/2+sqrt(2b*c*v-a*a/12)-halfspan
    elseif a+b ≤ c || v ≤ plic_corner_fraction(c, a, b, c)
        # Two clipped corners; use the real root of the cubic.
        radius = sqrt(2a*b)
        q = clamp(3*(a+b-2c*v)/(4radius), -one(T), one(T))
        offset = (a+b-c)/2-2radius*sin(asin(q)/3)
    else
        # Three clipped corners; evaluate the offset directly about the center.
        p = a*(b+c)+b*c-halfspan*halfspan
        radius = sqrt(p)
        q = clamp(3a*b*c*(one(T)/2-v)/(2p*radius), -one(T), one(T))
        offset = -2radius*sin(asin(q)/3)
    end
    (fraction < one(T)/2 ? offset : -offset)*scale
end

"""
    box_volume(n, α, lower, upper)

Measure of the region `n⋅x ≤ α` inside an axis-aligned rectangle or box:
area in 2D, volume in 3D. Coordinates use the same frame as `plane_fraction`.
This returns an absolute area or volume, not a fraction of the rectangle or
box. The calculation maps to a centered unit square or cube. Zero widths
return zero.
"""
function box_volume(n::Vec{N, T}, α::T, lower::Vec{N, T}, upper::Vec{N, T}) where {N, T<:AbstractFloat}
    N in (2, 3) || throw(ArgumentError("plane/box geometry supports two or three dimensions"))
    all(isfinite, lower) && all(isfinite, upper) || throw(ArgumentError("box corners must be finite"))
    widths = upper-lower
    all(≥(zero(T)), widths) || throw(ArgumentError("box widths must be nonnegative"))
    if any(iszero, widths)
        isnan(α) && throw(ArgumentError("the plane offset must not be NaN"))
        normal_scale(n)
        return zero(T)
    end
    center = lower/2+upper/2
    local_normal = n.*widths
    local_offset = α-sum(n.*center)
    prod(widths)*plane_fraction(local_normal, local_offset)
end

end # module PLIC
