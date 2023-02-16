struct uGIMP <: Kernel end

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


struct uGIMPValue{dim, T} <: MPValue{dim, T}
    itp::uGIMP
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function MPValue{dim, T}(itp::uGIMP) where {dim, T}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    uGIMPValue(itp, N, ∇N)
end

num_nodes(mp::uGIMPValue) = length(mp.N)
@inline shape_value(mp::uGIMPValue, j::Int) = (@_propagate_inbounds_meta; mp.N[j])
@inline shape_gradient(mp::uGIMPValue, j::Int) = (@_propagate_inbounds_meta; mp.∇N[j])

@inline function update_mpvalue!(mp::uGIMPValue, lattice::Lattice, pt)
    indices, _ = neighbornodes(mp.itp, lattice, pt)

    n = length(indices)
    resize!(mp.N, n)
    resize!(mp.∇N, n)

    @inbounds for (j, i) in enumerate(indices)
        mp.∇N[j], mp.N[j] = gradient(x->value(mp.itp,lattice,i,x,pt.l), getx(pt), :all)
    end

    indices
end
