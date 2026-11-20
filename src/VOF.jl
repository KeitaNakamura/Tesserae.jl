module VOF

using ..Tesserae: Vec
import ..PLIC
export myc_normal, myc_direction, flux

# Mixed Youngs-Centered reconstruction: Aulisa et al., JCP 225 (2007),
# 2301-2319, doi:10.1016/j.jcp.2007.03.015; Baraldi et al., Computers &
# Fluids 96 (2014), 322-337, Section 4.1. Selection follows the 2D/3D MYC
# conventions and zero-gradient guards used by Basilisk.
# https://basilisk.fr/src/myc2d.h ; https://basilisk.fr/src/myc.h

# Centered-column candidates: choose the most dominant normalized component.
@inline function myc_centered(field::AbstractArray{T, N}, I) where {T, N}
    sample(offset) = @inbounds field[I+offset]
    steps = ntuple(d -> CartesianIndex(ntuple(==(d), Val(N))), Val(N))

    central = ntuple(Val(N)) do d
        ed = steps[d]
        direction = sample(-ed)-sample(ed)
        for j in 1:N
            j == d && continue
            for side in (-1, 1)
                direction += sample(side*steps[j]-ed)-sample(side*steps[j]+ed)
            end
        end
        dominant = direction > 0 ? one(T) : -one(T)
        normal = Vec(ntuple(Val(N)) do j
            j == d && return dominant
            difference = zero(T)
            for side in -1:1
                difference += sample(side*ed-steps[j])-sample(side*ed+steps[j])
            end
            difference/2
        end)
        normal ./ sum(abs, normal)
    end
    best = N == 2 ? 2 : 1 # Match the reference tie-breaking order.
    for d in 1:N
        abs(central[d][d]) > abs(central[best][best]) && (best = d)
    end
    central[best], best
end

# Weighted negative gradient, with the standard 2D zero-gradient bias.
@inline function myc_youngs(field::AbstractArray{T, N}, center) where {T, N}
    sample(offset) = @inbounds field[center+offset]
    Vec(ntuple(Val(N)) do d
        difference = zero(T)
        # In 3D subtract opposite samples before weighting so symmetry gives exact zero.
        stencil = ntuple(j -> N == 3 && j == d ? (-1:-1) : (-1:1), Val(N))
        for I in CartesianIndices(stencil)
            iszero(I[d]) && continue
            weight = -I[d]
            for j in 1:N
                j != d && iszero(I[j]) && (weight *= 2)
            end
            value = N == 3 ? sample(I)-sample(Base.setindex(I, 1, d)) : sample(I)
            difference += T(weight)*value
        end
        N == 2 ? difference+T(1e-30) : difference
    end)
end

# Unconditional MYC estimate from finite samples, independent of phase handling.
function myc_direction(values::Vec{M, T}, ::Val{N}) where {M, T<:AbstractFloat, N}
    N in (2, 3) || throw(ArgumentError("MYC supports two or three dimensions"))
    M == 3^N || throw(DimensionMismatch("MYC requires a 3×3 or 3×3×3 neighborhood"))
    field = reshape(values, ntuple(_ -> 3, Val(N)))
    myc_direction(field, CartesianIndex(ntuple(_ -> 2, Val(N))))
end

"""
    myc_normal(fraction, I)

VOF interface normal at the Cartesian index `I` in a two- or three-dimensional
fraction array. Mixed cells read the surrounding 3×3 or 3×3×3 neighborhood and
return an outward `Vec` normal with unit L1 norm in reference coordinates. On
rectangular grids, `normal ./ spacing` gives its physical direction. Arbitrary
planar interfaces are not guaranteed to be exact.

The center must be finite. Empty/full cells (center ≤ 0 or ≥ 1) have no
interface in the cell and return the positive x convention without estimating
a normal from neighbors. Mixed cells require finite neighboring fractions.
Fractions are never changed or clipped; the transport scheme owns their bounds,
including roundoff. The caller supplies any required boundary values. Missing
neighbors raise BoundsError unless bounds checks are explicitly disabled. The
MYC zero-gradient guards use a 1e-30 component bias in 2D and a 1e-30 magnitude
cutoff in 3D.
"""
@inline function myc_normal(fraction::AbstractArray{T, N}, I::CartesianIndex{N}) where {T<:AbstractFloat, N}
    N in (2, 3) || throw(ArgumentError("MYC supports two or three dimensions"))
    center = fraction[I]
    isfinite(center) || throw(ArgumentError("the center fraction must be finite"))
    if center ≤ 0 || center ≥ 1
        return Vec(ntuple(d -> T(d==1), Val(N)))
    end
    myc_direction(fraction, I)
end

"""
    myc_direction(fraction, I)

MYC direction from the full 3×3 or 3×3×3 neighborhood of `I`, including when
the center is empty or full. Unlike [`myc_normal`](@ref), this always estimates
the direction from neighboring fractions. For example, a free-surface pressure
stencil can need that direction at a pure pressure cell adjacent to the interface.

The array must be two- or three-dimensional and the required neighbors must be
finite. The result has unit L1 norm in reference coordinates. The caller supplies
boundary values; missing neighbors raise BoundsError unless bounds checks are
explicitly disabled. Fractions are read without being changed or clipped.
"""
@inline function myc_direction(fraction::AbstractArray{T, N}, I::CartesianIndex{N}) where {T<:AbstractFloat, N}
    N in (2, 3) || throw(ArgumentError("MYC supports two or three dimensions"))
    neighborhood = CartesianIndices(ntuple(d -> (I[d]-1):(I[d]+1), Val(N)))
    @boundscheck checkbounds(fraction, neighborhood)
    for J in neighborhood
        isfinite(@inbounds fraction[J]) || throw(ArgumentError("neighboring fractions must be finite"))
    end
    candidate, axis = myc_centered(fraction, I)
    youngs = myc_youngs(fraction, I)
    magnitude = sum(abs, youngs)
    if N == 3 && magnitude < T(1e-30)
        return Vec(one(T), zero(T), zero(T))
    end
    iszero(magnitude) && return candidate
    youngs = youngs ./ magnitude
    use_youngs = if N == 2
        transverse = 3-axis
        abs(youngs[transverse])*abs(candidate[axis]) > abs(youngs[axis])*abs(candidate[transverse])
    else
        abs(candidate[axis]) > maximum(abs, youngs)
    end
    use_youngs ? youngs : candidate
end

"""
    flux(displacement, fraction, normal, offset, axis)
    flux(displacement, fraction_field, donor::CartesianIndex, axis)

Signed filled volume swept out of one donor cell, divided by its whole volume.
The caller selects the donor cell. `displacement = velocity*dt/spacing` chooses
its positive or negative face and must be finite with magnitude at most one.
The complete advection scheme can impose a stricter CFL bound (WY uses 1/2).

`normal` and `offset` describe the filled region in the centered unit cell.
The normal is a two- or three-dimensional `Vec`. Empty/full fractions use
`displacement*fraction`, retaining transport roundoff without clipping. The
caller monitors field bounds. Mixed cells use analytic plane/box geometry.
With a fraction field and donor index, use the same floating-point type for
`displacement` and the field elements. Mixed donors use MYC reconstruction
from one surrounding layer, including diagonals. The caller supplies the
halo when that neighborhood crosses a physical boundary. Pure donors need
only the center value.
"""
@inline function flux(displacement::T, fraction::T, normal::Vec{N, T}, offset::T, axis::Integer) where {N, T<:AbstractFloat}
    check_flux(displacement, axis, N)
    isfinite(fraction) || throw(ArgumentError("the donor fraction must be finite"))
    iszero(displacement) && return zero(T)
    (fraction ≤ 0 || fraction ≥ 1) && return displacement*fraction
    geometric_flux(displacement, normal, offset, axis)
end

@inline function flux(displacement::T, fraction::AbstractArray{T, N}, donor::CartesianIndex{N}, axis::Integer) where {N, T<:AbstractFloat}
    check_flux(displacement, axis, N)
    c = fraction[donor]
    isfinite(c) || throw(ArgumentError("the donor fraction must be finite"))
    iszero(displacement) && return zero(T)
    (c ≤ 0 || c ≥ 1) && return displacement*c
    reconstructed_flux(displacement, fraction, donor, c, axis)
end

@inline function check_flux(displacement, axis, N)
    N in (2, 3) || throw(ArgumentError("the donor cell must be two- or three-dimensional"))
    1 ≤ axis ≤ N || throw(ArgumentError("invalid flux axis"))
    isfinite(displacement) && abs(displacement) ≤ 1 || throw(ArgumentError("displacement must lie in [-1,1]"))
end

@noinline function reconstructed_flux(displacement, fraction, donor, c, axis)
    normal = myc_direction(fraction, donor)
    geometric_flux(displacement, normal, PLIC.plane_offset(normal, c), axis)
end

@inline function geometric_flux(displacement::T, normal::Vec{N, T}, offset, axis) where {N, T}
    # Map the swept slab to the unit cell, keeping its given width.
    slab_normal = Vec(ntuple(d -> d == axis ? abs(displacement)*normal[d] : normal[d], Val(N)))
    face_offset = offset-sign(displacement)*normal[axis]/2
    slab_offset = face_offset+displacement*normal[axis]/2
    displacement*PLIC.plane_fraction(slab_normal, slab_offset)
end

end # module VOF
