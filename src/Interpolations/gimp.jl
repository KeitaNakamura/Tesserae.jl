struct GIMP <: Kernel end

@pure getnnodes(f::GIMP, ::Val{dim}) where {dim} = prod(nfill(3, Val(dim)))

@inline function neighbornodes(f::GIMP, grid::Grid, xp::Vec, rp::Vec)
    dx⁻¹ = gridsteps_inv(grid)
    neighbornodes(grid, xp, 1 .+ rp.*dx⁻¹)
end
@inline neighbornodes(f::GIMP, grid::Grid, pt) = neighbornodes(f, grid, pt.x, pt.r)

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
@inline value(f::GIMP, ξ::Vec, l::Vec) = prod(maptuple(value, f, Tuple(ξ), Tuple(l)))
# used in `WLS`
function value(f::GIMP, grid::Grid, I::Index, xp::Vec, rp::Vec)
    @_inline_propagate_inbounds_meta
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    value(f, ξ, rp.*dx⁻¹)
end
@inline value(f::GIMP, grid::Grid, I::Index, pt) = value(f, grid, I, pt.x, pt.r)
# used in `KernelCorrection`
function value_gradient(f::GIMP, grid::Grid, I::Index, xp::Vec, rp::Vec)
    @_inline_propagate_inbounds_meta
    xi = grid[I]
    dx⁻¹ = gridsteps_inv(grid)
    ξ = (xp - xi) .* dx⁻¹
    ∇w, w = gradient(ξ -> value(f, ξ, rp.*dx⁻¹), ξ, :all)
    w, ∇w.*dx⁻¹
end
@inline value_gradient(f::GIMP, grid::Grid, I::Index, pt) = value_gradient(f, grid, I, pt.x, pt.r)

# used in `WLS`
# `x` and `l` must be normalized by `dx`
@inline function Base.values(::GIMP, x::T, l::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    maptuple(value, GIMP(), Tuple(ξ), Tuple(l))
end
@inline Base.values(f::GIMP, x::Vec, l::Vec) = Tuple(otimes(maptuple(values, f, Tuple(x), Tuple(l))...))
function Base.values(f::GIMP, grid::Grid, xp::Vec, lp::Vec)
    dx⁻¹ = gridsteps_inv(grid)
    values(f, xp.*dx⁻¹, lp.*dx⁻¹)
end
@inline Base.values(f::GIMP, grid::Grid, pt) = values(f, grid, pt.x, pt.r)

# used in `KernelCorrection`
# `x` and `l` must be normalized by `dx`
_gradient_GIMP(x, l) = gradient(x -> value(GIMP(), x, l), x, :all)
function _values_gradients(::GIMP, x::T, l::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    vals_grads = maptuple(_gradient_GIMP, Tuple(ξ), l)
    vals  = maptuple(getindex, vals_grads, 2)
    grads = maptuple(getindex, vals_grads, 1)
    Vec(vals), Vec(grads)
end
@generated function values_gradients(::GIMP, x::Vec{dim}, l::Vec{dim}) where {dim}
    exps = map(1:dim) do i
        x = [d == i ? :(grads[$d]) : :(vals[$d]) for d in 1:dim]
        :(Tuple(otimes($(x...))))
    end
    quote
        @_inline_meta
        vals_grads = maptuple(_values_gradients, GIMP(), Tuple(x), Tuple(l))
        vals  = maptuple(getindex, vals_grads, 1)
        grads = maptuple(getindex, vals_grads, 2)
        Tuple(otimes(vals...)), maptuple(Vec, $(exps...))
    end
end
function values_gradients(f::GIMP, grid::Grid, xp::Vec, lp::Vec)
    dx⁻¹ = gridsteps_inv(grid)
    wᵢ, ∇wᵢ = values_gradients(f, xp.*dx⁻¹, lp.*dx⁻¹)
    wᵢ, broadcast(.*, ∇wᵢ, Ref(dx⁻¹))
end
@inline values_gradients(f::GIMP, grid::Grid, pt) = values_gradients(f, grid, pt.x, pt.r)


struct GIMPValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    xp::Vec{dim, T}
end

mutable struct GIMPValues{dim, T, L} <: MPValues{dim, T, GIMPValue{dim, T}}
    F::GIMP
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    gridindices::MVector{L, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

# constructors
function GIMPValues{dim, T, L}() where {dim, T, L}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    gridindices = MVector{L, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    GIMPValues(GIMP(), N, ∇N, gridindices, xp, 0)
end
function MPValues{dim, T}(F::GIMP) where {dim, T}
    L = getnnodes(F, Val(dim))
    GIMPValues{dim, T, L}()
end

getkernelfunction(x::GIMPValues) = x.F

function update!(mpvalues::GIMPValues, grid::Grid, pt, spat::AbstractArray{Bool})
    # reset
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)

    F = getkernelfunction(mpvalues)
    xp = pt.x

    # update
    mpvalues.xp = xp
    dx⁻¹ = gridsteps_inv(grid)
    update_active_gridindices!(mpvalues, neighbornodes(F, grid, pt), spat)
    @inbounds @simd for i in 1:length(mpvalues)
        I = gridindices(mpvalues, i)
        mpvalues.N[i], mpvalues.∇N[i] = value_gradient(F, grid, I, pt)
    end
    mpvalues
end

@inline function Base.getindex(mpvalues::GIMPValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.xp)
end
