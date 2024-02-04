"""
    uGIMP()

The unchanged GIMP kernel [^uGIMP].

[^uGIMP]: [Bardenhagen, S. G., & Kober, E. M. (2004). The generalized interpolation material point method. *Computer Modeling in Engineering and Sciences*, 5(6), 477-496.](https://doi.org/10.3970/cmes.2004.005.477)
"""
struct uGIMP <: Kernel end

gridspan(::uGIMP) = 3

@inline function neighbornodes(::uGIMP, lattice::Lattice, pt)
    dx⁻¹ = spacing_inv(lattice)
    neighbornodes(lattice, getx(pt), 1+(pt.l/2)*dx⁻¹)
end

# simple uGIMP calculation
# See Eq.(40) in
# Bardenhagen, S. G., & Kober, E. M. (2004).
# The generalized interpolation material point method.
# Computer Modeling in Engineering and Sciences, 5(6), 477-496.
# boundary treatment is ignored
function value(::uGIMP, ξ::Real, l::Real) # `l` is normalized radius by dx
    ξ = abs(ξ)
    ξ < l/2   ? 1 - (4ξ^2 + l^2) / 4l :
    ξ < 1-l/2 ? 1 - ξ                 :
    ξ < 1+l/2 ? (2+l-2ξ)^2 / 8l       : zero(ξ)
end
@inline value(f::uGIMP, ξ::Vec, l::Real) = prod(value.(f, ξ, l))
@inline function value(f::uGIMP, lattice::Lattice, I::CartesianIndex, xₚ::Vec, lₚ::Real)
    @_propagate_inbounds_meta
    xᵢ = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xₚ - xᵢ) * dx⁻¹
    value(f, ξ, lₚ*dx⁻¹)
end
@inline value(f::uGIMP, lattice::Lattice, I::CartesianIndex, pt) = value(f, lattice, I, getx(pt), pt.l)

@inline function value_gradient(f::uGIMP, lattice::Lattice, I::CartesianIndex, pt)
    @_propagate_inbounds_meta
    ∇N, N = gradient(x -> (@_propagate_inbounds_meta; value(f, lattice, I, x, pt.l)), getx(pt), :all)
    N, ∇N
end

@inline function update_property!(mp::MPValues{uGIMP}, lattice::Lattice, pt)
    indices = neighbornodes(mp)
    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        mp.N[ip], mp.∇N[ip] = value_gradient(interpolation(mp), lattice, i, pt)
    end
end
