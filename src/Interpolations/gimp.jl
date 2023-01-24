struct GIMP <: Kernel end

@pure num_nodes(f::GIMP, ::Val{dim}) where {dim} = prod(nfill(3, Val(dim)))

@inline function nodeindices(f::GIMP, grid::Grid, xp::Vec, rp::Vec)
    dx⁻¹ = gridsteps_inv(grid)
    nodeindices(grid, xp, 1 .+ rp.*dx⁻¹)
end
@inline nodeindices(f::GIMP, grid::Grid, pt) = nodeindices(f, grid, pt.x, pt.r)

# simple GIMP calculation
# See Eq.(40) in
# Bardenhagen, S. G., & Kober, E. M. (2004).
# The generalized interpolation material point method.
# Computer Modeling in Engineering and Sciences, 5(6), 477-496.
# boundary treatment is ignored
function value(::GIMP, ξ::Real, l::Real) # `l` is normalized radius
    ξ = abs(ξ)
    ξ < l   ? 1 - (ξ^2 + l^2) / 2l :
    ξ < 1-l ? 1 - ξ                :
    ξ < 1+l ? (1+l-ξ)^2 / 4l       : zero(ξ)
end
@inline value(f::GIMP, ξ::Vec, l::Vec) = prod(value.(f, ξ, l))
function value(f::GIMP, grid::Grid, I::Index, xp::Vec, rp::Vec)
    @_propagate_inbounds_meta
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    value(f, ξ, rp.*dx⁻¹)
end
@inline value(f::GIMP, grid::Grid, I::Index, pt) = value(f, grid, I, pt.x, pt.r)

# `x` and `l` must be normalized by `dx`
_gradient_GIMP(x, l) = gradient(x -> value(GIMP(), x, l), x, :all)
function _values_gradients(::GIMP, x::T, l::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    vals_grads = _gradient_GIMP.(Tuple(ξ), Tuple(l))
    vals  = getindex.(vals_grads, 2)
    grads = getindex.(vals_grads, 1)
    Vec(vals), Vec(grads)
end
@generated function values_gradients(::GIMP, x::Vec{dim}, l::Vec{dim}) where {dim}
    quote
        vals_grads = @ntuple $dim d -> _values_gradients(GIMP(), x[d], l[d])
        vals  = getindex.(vals_grads, 1)
        grads = getindex.(vals_grads, 2)
        Tuple(otimes(vals...)), Vec.((@ntuple $dim i -> begin
                                          Tuple(otimes((@ntuple $dim d -> d==i ? grads[d] : vals[d])...))
                                      end)...)
    end
end
function values_gradients(f::GIMP, grid::Grid, xp::Vec, lp::Vec)
    dx⁻¹ = gridsteps_inv(grid)
    wᵢ, ∇wᵢ = values_gradients(f, xp.*dx⁻¹, lp.*dx⁻¹)
    wᵢ, broadcast(.*, ∇wᵢ, Ref(dx⁻¹))
end
values_gradients(f::GIMP, grid::Grid, pt) = values_gradients(f, grid, pt.x, pt.r)


mutable struct GIMPValue{dim, T, L} <: MPValue{dim, T}
    F::GIMP
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    # necessary in MPValue
    nodeindices::MVector{L, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

function MPValue{dim, T}(F::GIMP) where {dim, T}
    L = num_nodes(F, Val(dim))
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    nodeindices = MVector{L, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    GIMPValue(F, N, ∇N, nodeindices, xp, 0)
end

get_kernel(mp::GIMPValue) = mp.F
@inline function mpvalue(mp::GIMPValue, i::Int)
    @boundscheck @assert 1 ≤ i ≤ num_nodes(mp)
    (; N=mp.N[i], ∇N=mp.∇N[i], xp=mp.xp)
end

function update_kernels!(mp::GIMPValue, grid::Grid, pt)
    # reset
    fillzero!(mp.N)
    fillzero!(mp.∇N)
    # update
    F = get_kernel(mp)
    @inbounds @simd for i in 1:num_nodes(mp)
        I = nodeindex(mp, i)
        mp.∇N[i], mp.N[i] = gradient(x->value(F,grid,I,x,pt.r), pt.x, :all)
    end
    mp
end
