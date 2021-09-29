struct Polynomial{order}
    function Polynomial{order}() where {order}
        new{order::Int}()
    end
end

# value
value(poly::Polynomial{0}, x::Vec) = Vec(one(eltype(x)))
value(poly::Polynomial{1}, x::Vec{1, T}) where {T} = @inbounds Vec(one(T), x[1])
value(poly::Polynomial{1}, x::Vec{2, T}) where {T} = @inbounds Vec(one(T), x[1], x[2])
value(poly::Polynomial{1}, x::Vec{3, T}) where {T} = @inbounds Vec(one(T), x[1], x[2], x[3])
# gradient
function Tensorial.gradient(poly::Polynomial{1}, x::Vec{1, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z
          o]
end
function Tensorial.gradient(poly::Polynomial{1}, x::Vec{2, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z z
          o z
          z o]
end
function Tensorial.gradient(poly::Polynomial{1}, x::Vec{3, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z z z
          o z z
          z o z
          z z o]
end

# for ∇ operation
struct PolynomialGradient{order}
    parent::Polynomial{order}
end
Base.adjoint(p::Polynomial{order}) where {order} = PolynomialGradient(p)

# function like methods
(p::Polynomial)(x) = value(p, x)
(p::PolynomialGradient)(x) = gradient(p.parent, x)


struct WLS{poly_order, bspline_order} <: ShapeFunction
    poly::Polynomial{poly_order}
    bspline::BSpline{bspline_order}
end

const LinearWLS{bspline_order} = WLS{1, bspline_order}

WLS{poly_order, bspline_order}() where {poly_order, bspline_order} = WLS(Polynomial{poly_order}(), BSpline{bspline_order}())
WLS{poly_order}(bspline::BSpline) where {poly_order} = WLS(Polynomial{poly_order}(), bspline)

polynomial(wls::WLS) = wls.poly
weight_function(wls::WLS) = wls.bspline

support_length(wls::WLS) = support_length(weight_function(wls))


struct WLSValues{poly_order, bspline_order, dim, T, L, M, O} <: ShapeValues{dim, T}
    F::WLS{poly_order, bspline_order}
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    w::MVector{L, T}
    M⁻¹::Base.RefValue{Mat{M, M, T, O}}
    inds::MVector{L, Index{dim}}
    len::Base.RefValue{Int}
end

polynomial(it::WLSValues) = polynomial(it.F)
weight_function(it::WLSValues) = weight_function(it.F)

function WLSValues{poly_order, bspline_order, dim, T, L, M, O}() where {poly_order, bspline_order, dim, T, L, M, O}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    w = MVector{L, T}(undef)
    M⁻¹ = zero(Mat{M, M, T, O})
    inds = MVector{L, Index{dim}}(undef)
    WLSValues(WLS{poly_order, bspline_order}(), N, ∇N, w, Ref(M⁻¹), inds, Ref(0))
end

function ShapeValues{dim, T}(F::WLS{poly_order, bspline_order}) where {poly_order, bspline_order, dim, T}
    p = polynomial(F)
    M = length(p(zero(Vec{dim, T})))
    L = nnodes(weight_function(F), Val(dim))
    WLSValues{poly_order, bspline_order, dim, T, L, M, M^2}()
end

function update!(it::WLSValues{<: Any, <: Any, dim}, grid, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    update_gridindices!(it, grid, x, spat)
    it.N .= zero(it.N)
    it.∇N .= zero(it.∇N)
    it.w .= zero(it.w)
    F = weight_function(it)
    P = polynomial(it)
    M = zero(it.M⁻¹[])
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) ./ gridsteps(grid)
        it.w[i] = value(F, ξ)
    end
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        p = P(xᵢ - x)
        M += it.w[i] * p ⊗ p
    end
    it.M⁻¹[] = inv(M)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        q = it.M⁻¹[] ⋅ P(xᵢ - x)
        wq = it.w[i] * q
        it.N[i] = wq ⋅ P(x - x)
        it.∇N[i] = wq ⋅ P'(x - x)
    end
    it
end


struct WLSValue{dim, T, L, M}
    N::T
    ∇N::Vec{dim, T}
    w::T
    M⁻¹::Mat{L, L, T, M}
    index::Index{dim}
end

@inline function Base.getindex(it::WLSValues, i::Int)
    @_propagate_inbounds_meta
    WLSValue(it.N[i], it.∇N[i], it.w[i], it.M⁻¹[], it.inds[i])
end
