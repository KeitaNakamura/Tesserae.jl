"""
    GIMP()

The unchanged GIMP (generalized interpolation material point) kernel [^GIMP].

[^GIMP]: [Bardenhagen, S. G., & Kober, E. M. (2004). The generalized interpolation material point method. *Computer Modeling in Engineering and Sciences*, 5(6), 477-496.](https://doi.org/10.3970/cmes.2004.005.477)
"""
struct GIMP <: Kernel end

gridspan(::GIMP) = 3

@inline function neighboringnodes(::GIMP, pt, mesh::CartesianMesh)
    dx⁻¹ = spacing_inv(mesh)
    neighboringnodes(getx(pt), 1+pt.l*dx⁻¹, mesh)
end

# simple GIMP calculation
# See Eq.(40) in
# Bardenhagen, S. G., & Kober, E. M. (2004).
# The generalized interpolation material point method.
# Computer Modeling in Engineering and Sciences, 5(6), 477-496.
# boundary treatment is ignored
function value(::GIMP, ξ::Real, l::Real) # `2l` is the particle size normalized by dx
    ξ = abs(ξ)
    ξ < l   ? 1 - (ξ^2 + l^2) / 2l :
    ξ < 1-l ? 1 - ξ                :
    ξ < 1+l ? (1+l-ξ)^2 / 4l       : zero(ξ)
end
@inline value(f::GIMP, ξ::Vec, l::Real) = prod(value.((f,), ξ, l))
@inline function value(f::GIMP, xₚ::Vec, lₚ::Real, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    xᵢ = mesh[I]
    dx⁻¹ = spacing_inv(mesh)
    ξ = (xₚ - xᵢ) * dx⁻¹
    value(f, ξ, lₚ*dx⁻¹)
end
@inline value(f::GIMP, pt, mesh::CartesianMesh, I::CartesianIndex) = value(f, getx(pt), pt.l, mesh, I)

@inline function value_gradient(f::GIMP, pt, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    ∇N, N = gradient(x -> (@_propagate_inbounds_meta; value(f, x, pt.l, mesh, I)), getx(pt), :all)
    N, ∇N
end

@inline function update_property!(mp::MPValues{GIMP}, pt, mesh::CartesianMesh)
    indices = neighboringnodes(mp)
    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        mp.N[ip], mp.∇N[ip] = value_gradient(interpolation(mp), pt, mesh, i)
    end
end
