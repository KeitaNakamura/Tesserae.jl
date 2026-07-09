"""
    uGIMP()

A kernel for the unchanged generalized interpolation material point (uGIMP) [^GIMP].
`uGIMP` requires the initial particle length `l` in the particle property as follows:

```jl
ParticleProp = @NamedTuple begin
    < variables... >
    l :: Float64
end
```

[^GIMP]: [Bardenhagen, S. G., & Kober, E. M. (2004). The generalized interpolation material point method. *Computer Modeling in Engineering and Sciences*, 5(6), 477-496.](https://doi.org/10.3970/cmes.2004.005.477)
"""
struct uGIMP <: Kernel end

support_width(::uGIMP) = 3

@inline function supportnodes(::uGIMP, pt, mesh::CartesianMesh)
    h⁻¹ = spacing_inv(mesh)
    supportnodes(getx(pt), 1+pt.l*h⁻¹, mesh)
end

# simple uGIMP calculation
# See Eq.(40) in
# Bardenhagen, S. G., & Kober, E. M. (2004).
# The generalized interpolation material point method.
# Computer Modeling in Engineering and Sciences, 5(6), 477-496.
# boundary treatment is ignored
@inline function value(::uGIMP, ξ::Real, l::Real) # `l` is the particle size normalized by h
    ξ = abs(ξ)
    ξ < l/2   ? 1 - (4ξ^2+l^2)/4l :
    ξ < 1-l/2 ? 1 - ξ             :
    ξ < 1+l/2 ? (1+l/2-ξ)^2 / 2l  : zero(ξ)
end

@inline function jet(::Order{k}, gimp::uGIMP, ξ::Real, l::Real) where {k}
    reverse(∂{k}(ξ -> value(gimp, ξ, l), ξ, :all))
end

@generated function basis_jet(order::Order{k}, spline::uGIMP, pt, mesh::CartesianMesh{dim}, i) where {dim, k}
    quote
        @_inline_meta
        x = getx(pt)
        h⁻¹ = spacing_inv(mesh)
        ξ = (x - mesh[i]) * h⁻¹
        l = pt.l * h⁻¹
        vals′ = @ntuple $dim d -> jet(order, spline, ξ[d], l)
        vals = @ntuple $(k+1) a -> only(prod_each_dimension(Order(a-1), vals′...))
        @ntuple $(k+1) i -> vals[i]*h⁻¹^(i-1)
    end
end

@inline function update_basis_values!(bw::BasisWeight, gimp::uGIMP, pt, mesh::CartesianMesh)
    indices = supportnodes(bw)
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        set_values!(bw, ip, basis_jet(derivative_order(bw), gimp, pt, mesh, i))
    end
end
