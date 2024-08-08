"""
    uGIMP()

The unchanged GIMP (generalized interpolation material point) kernel [^GIMP].

[^GIMP]: [Bardenhagen, S. G., & Kober, E. M. (2004). The generalized interpolation material point method. *Computer Modeling in Engineering and Sciences*, 5(6), 477-496.](https://doi.org/10.3970/cmes.2004.005.477)
"""
struct uGIMP <: Kernel end

get_kernel(k::uGIMP) = k
gridspan(::uGIMP) = 3

@inline function neighboringnodes(::uGIMP, pt, mesh::CartesianMesh)
    h⁻¹ = spacing_inv(mesh)
    neighboringnodes(getx(pt), 1+pt.l*h⁻¹, mesh)
end

# simple uGIMP calculation
# See Eq.(40) in
# Bardenhagen, S. G., & Kober, E. M. (2004).
# The generalized interpolation material point method.
# Computer Modeling in Engineering and Sciences, 5(6), 477-496.
# boundary treatment is ignored
function value(::uGIMP, ξ::Real, l::Real) # `2l` is the particle size normalized by h
    ξ = abs(ξ)
    ξ < l   ? 1 - (ξ^2 + l^2) / 2l :
    ξ < 1-l ? 1 - ξ                :
    ξ < 1+l ? (1+l-ξ)^2 / 4l       : zero(ξ)
end
@inline value(f::uGIMP, ξ::Vec, l::Real) = prod(value.((f,), ξ, l))
@inline function value(f::uGIMP, xₚ::Vec, lₚ::Real, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    xᵢ = mesh[I]
    h⁻¹ = spacing_inv(mesh)
    ξ = (xₚ - xᵢ) * h⁻¹
    value(f, ξ, lₚ*h⁻¹)
end
@inline value(f::uGIMP, pt, mesh::CartesianMesh, I::CartesianIndex) = value(f, getx(pt), pt.l, mesh, I)

@inline function value(::Order{1}, gimp::uGIMP, pt, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    reverse(gradient(x -> (@_propagate_inbounds_meta; value(gimp, x, pt.l, mesh, I)), getx(pt), :all))
end
@inline function value(::Order{2}, gimp::uGIMP, pt, mesh::CartesianMesh, I::CartesianIndex)
    @_propagate_inbounds_meta
    reverse(hessian(x -> (@_propagate_inbounds_meta; value(gimp, x, pt.l, mesh, I)), getx(pt), :all))
end

@inline function update_property!(mp::MPValue, it::uGIMP, pt, mesh::CartesianMesh)
    indices = neighboringnodes(mp)
    @inbounds @simd for ip in eachindex(indices)
        i = indices[ip]
        set_kernel_values!(mp, ip, value(derivative_order(mp), it, pt, mesh, i))
    end
end
