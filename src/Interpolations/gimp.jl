"""
    uGIMP()

The unchanged GIMP (generalized interpolation material point) kernel [^uGIMP].

[^uGIMP]: [Bardenhagen, S. G., & Kober, E. M. (2004). The generalized interpolation material point method. *Computer Modeling in Engineering and Sciences*, 5(6), 477-496.](https://doi.org/10.3970/cmes.2004.005.477)
"""
struct uGIMP <: Kernel end

gridspan(::uGIMP) = 3

@inline function surroundingnodes(::uGIMP, pt, mesh::CartesianMesh)
    dx⁻¹ = spacing_inv(mesh)
    surroundingnodes(getx(pt), 1+(pt.l/2)*dx⁻¹, mesh)
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
@inline function value(f::uGIMP, xₚ::Vec, lₚ::Real, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    xᵢ = mesh[I]
    dx⁻¹ = spacing_inv(mesh)
    ξ = (xₚ - xᵢ) * dx⁻¹
    value(f, ξ, lₚ*dx⁻¹)
end
@inline value(f::uGIMP, pt, mesh::CartesianMesh, I::CartesianIndex) = value(f, getx(pt), pt.l, mesh, I)

@inline function value_gradient(f::uGIMP, pt, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    ∇N, N = gradient(x -> (@_propagate_inbounds_meta; value(f, x, pt.l, mesh, I)), getx(pt), :all)
    N, ∇N
end

@inline function update_property!(mp::MPValues{uGIMP}, pt, mesh::CartesianMesh)
    indices = surroundingnodes(mp)
    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        mp.N[ip], mp.∇N[ip] = value_gradient(interpolation(mp), pt, mesh, i)
    end
end
