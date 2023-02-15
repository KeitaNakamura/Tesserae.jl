struct GIMP <: Kernel end

@inline function neighbornodes(::GIMP, lattice::Lattice, pt)
    dx⁻¹ = spacing_inv(lattice)
    neighbornodes(lattice, pt.x, 1+(pt.l/2)*dx⁻¹)
end

# simple GIMP calculation
# See Eq.(40) in
# Bardenhagen, S. G., & Kober, E. M. (2004).
# The generalized interpolation material point method.
# Computer Modeling in Engineering and Sciences, 5(6), 477-496.
# boundary treatment is ignored
function value(::GIMP, ξ::Real, l::Real) # `l` is normalized radius by dx
    ξ = abs(ξ)
    ξ < l/2   ? 1 - (4ξ^2 + l^2) / 4l :
    ξ < 1-l/2 ? 1 - ξ                 :
    ξ < 1+l/2 ? (2+l-2ξ)^2 / 8l       : zero(ξ)
end
@inline value(f::GIMP, ξ::Vec, l::Real) = prod(value.(f, ξ, l))
function value(f::GIMP, lattice::Lattice, I::CartesianIndex, xp::Vec, lp::Real)
    @_propagate_inbounds_meta
    xi = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xp - xi) * dx⁻¹
    value(f, ξ, lp*dx⁻¹)
end
@inline value(f::GIMP, lattice::Lattice, I::CartesianIndex, pt) = value(f, lattice, I, pt.x, pt.l)

@inline function value_gradient(f::GIMP, lattice::Lattice, I::CartesianIndex, pt)
    ∇N, N = gradient(x -> value(f, lattice, I, x, pt.l), pt.x, :all)
    N, ∇N
end


struct GIMPValue{dim, T} <: MPValue{dim, T}
    itp::GIMP
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function MPValue{dim, T}(itp::GIMP) where {dim, T}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    GIMPValue(itp, N, ∇N)
end

num_nodes(mp::GIMPValue) = length(mp.N)
@inline shape_value(mp::GIMPValue, j::Int) = (@_propagate_inbounds_meta; mp.N[j])
@inline shape_gradient(mp::GIMPValue, j::Int) = (@_propagate_inbounds_meta; mp.∇N[j])

@inline function update_mpvalue!(mp::GIMPValue, lattice::Lattice, pt)
    indices, _ = neighbornodes(mp.itp, lattice, pt)

    n = length(indices)
    resize!(mp.N, n)
    resize!(mp.∇N, n)

    @inbounds for (j, i) in enumerate(indices)
        mp.∇N[j], mp.N[j] = gradient(x->value(mp.itp,lattice,i,x,pt.l), getx(pt), :all)
    end

    indices
end
