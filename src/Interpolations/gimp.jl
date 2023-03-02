struct uGIMP <: Kernel end

gridsize(::uGIMP) = 3

@inline function neighbornodes(::uGIMP, lattice::Lattice, pt)
    dx⁻¹ = spacing_inv(lattice)
    neighbornodes(lattice, pt.x, 1+(pt.l/2)*dx⁻¹)
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
function value(f::uGIMP, lattice::Lattice, I::CartesianIndex, xp::Vec, lp::Real)
    @_propagate_inbounds_meta
    xi = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xp - xi) * dx⁻¹
    value(f, ξ, lp*dx⁻¹)
end
@inline value(f::uGIMP, lattice::Lattice, I::CartesianIndex, pt) = value(f, lattice, I, pt.x, pt.l)

@inline function value_gradient(f::uGIMP, lattice::Lattice, I::CartesianIndex, pt)
    ∇N, N = gradient(x -> value(f, lattice, I, x, pt.l), pt.x, :all)
    N, ∇N
end

function MPValuesInfo{dim, T}(itp::uGIMP) where {dim, T}
    dims = nfill(gridsize(itp), Val(dim))
    values = (; N=zero(T), ∇N=zero(Vec{dim, T}))
    sizes = (dims, dims)
    MPValuesInfo{dim, T}(values, sizes)
end

@inline function update_mpvalues!(mp::MPValues, itp::uGIMP, lattice::Lattice, pt)
    indices, _ = neighbornodes(itp, lattice, pt)

    @inbounds for (j, i) in pairs(IndexCartesian(), indices)
        mp.∇N[j], mp.N[j] = gradient(x->value(itp,lattice,i,x,pt.l), getx(pt), :all)
    end

    indices
end
