struct GIMP <: Kernel end

@pure maxnum_nodes(f::GIMP, ::Val{dim}) where {dim} = prod(nfill(3, Val(dim)))

@inline function neighbornodes(f::GIMP, lattice::Lattice, xp::Vec, rp::Real)
    dx⁻¹ = spacing_inv(lattice)
    neighbornodes(lattice, xp, 1 + rp*dx⁻¹)
end
@inline neighbornodes(f::GIMP, lattice::Lattice, pt) = neighbornodes(f, lattice, pt.x, pt.r)

# simple GIMP calculation
# See Eq.(40) in
# Bardenhagen, S. G., & Kober, E. M. (2004).
# The generalized interpolation material point method.
# Computer Modeling in Engineering and Sciences, 5(6), 477-496.
# boundary treatment is ignored
function value(::GIMP, ξ::Real, l::Real) # `l` is normalized radius by dx
    ξ = abs(ξ)
    ξ < l   ? 1 - (ξ^2 + l^2) / 2l :
    ξ < 1-l ? 1 - ξ                :
    ξ < 1+l ? (1+l-ξ)^2 / 4l       : zero(ξ)
end
@inline value(f::GIMP, ξ::Vec, l::Real) = prod(value.(f, ξ, l))
function value(f::GIMP, lattice::Lattice, I::CartesianIndex, xp::Vec, rp::Real)
    @_propagate_inbounds_meta
    xi = lattice[I]
    dx⁻¹ = spacing_inv(lattice)
    ξ = (xp - xi) * dx⁻¹
    value(f, ξ, rp*dx⁻¹)
end
@inline value(f::GIMP, lattice::Lattice, I::CartesianIndex, pt) = value(f, lattice, I, pt.x, pt.r)

# `x` and `l` must be normalized by `dx`
_gradient_GIMP(x, l) = gradient(x -> value(GIMP(), x, l), x, :all)
function _values_gradients(::GIMP, x::T, l::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    vals_grads = _gradient_GIMP.(Tuple(ξ), l)
    vals  = getindex.(vals_grads, 2)
    grads = getindex.(vals_grads, 1)
    Vec(vals), Vec(grads)
end
@generated function values_gradients(::GIMP, lattice::Lattice{dim}, xp::Vec{dim}, lp::Real) where {dim}
    quote
        @_inline_meta
        dx⁻¹ = spacing_inv(lattice)
        x = (xp - first(lattice)) * dx⁻¹
        vals_grads = @ntuple $dim d -> _values_gradients(GIMP(), x[d], lp*dx⁻¹)
        vals  = getindex.(vals_grads, 1)
        grads = getindex.(vals_grads, 2) .* dx⁻¹
        Tuple(otimes(vals...)), Vec.((@ntuple $dim i -> begin
                                          Tuple(otimes((@ntuple $dim d -> d==i ? grads[d] : vals[d])...))
                                      end)...)
    end
end
values_gradients(f::GIMP, lattice::Lattice, pt) = values_gradients(f, lattice, pt.x, pt.r)


struct GIMPValue{dim, T} <: MPValue{dim, T, GIMP}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function MPValue{dim, T}(::GIMP) where {dim, T}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    GIMPValue{dim, T}(N, ∇N)
end

@inline function update!(mp::GIMPValue, ::NearBoundary, lattice::Lattice, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::CartesianIndices, pt)
    n = length(nodeinds)
    F = get_kernel(mp)
    resize!(mp.N, n)
    resize!(mp.∇N, n)
    @inbounds for (j, i) in enumerate(nodeinds)
        mp.∇N[j], mp.N[j] = gradient(x->value(F,lattice,i,x,pt.r), pt.x, :all) .* sppat[i]
    end
    mp
end
