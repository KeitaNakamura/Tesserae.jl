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
    l = _normalized_particle_length(pt, mesh)
    zero(l) ≤ l ≤ one(l) || throw(ArgumentError("uGIMP requires 0 ≤ pt.l ≤ spacing(mesh)"))
    supportnodes(getx(pt), one(l) + l/2, mesh)
end
@inline _normalized_particle_length(pt, mesh::CartesianMesh) = pt.l / spacing(mesh)

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

# Every axis sees the same normalized particle length.
@inline axis_jet_args(::uGIMP, pt, mesh::CartesianMesh{dim}, i) where {dim} =
    nfill((_normalized_particle_length(pt, mesh),), Val(dim))

@inline nodal_basis_jet(order::Order, spline::uGIMP, pt, mesh::CartesianMesh, i) =
    separable_nodal_basis_jet(order, spline, pt, mesh, i)
