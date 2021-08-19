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


struct WLS{poly_order, bspline_order, dim} <: ShapeFunction{dim}
    poly::Polynomial{poly_order}
    bspline::BSpline{bspline_order, dim}
end

WLS{poly_order, bspline_order, dim}() where {poly_order, bspline_order, dim} =
    WLS(Polynomial{poly_order}(), BSpline{bspline_order, dim}())

WLS{poly_order}(bspline::BSpline) where {poly_order} = WLS(Polynomial{poly_order}(), bspline)

polynomial(wls::WLS) = wls.poly
weight_function(wls::WLS) = wls.bspline

support_length(wls::WLS) = support_length(weight_function(wls))


struct WLSValues{poly_order, bspline_order, dim, T, L, M} <: ShapeValues{dim, T}
    F::WLS{poly_order, bspline_order, dim}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
    w::Vector{T}
    M⁻¹::Base.RefValue{Mat{L, L, T, M}}
end

polynomial(it::WLSValues) = polynomial(it.F)
weight_function(it::WLSValues) = weight_function(it.F)

weight_value(it::WLSValues) = Collection{1}(it.w)
moment_matrix_inverse(it::WLSValues) = it.M⁻¹[]

function WLSValues{poly_order, bspline_order, dim, T, L, M}() where {poly_order, bspline_order, dim, T, L, M}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    w = Vector{T}(undef, 0)
    M⁻¹ = zero(Mat{L, L, T, M})
    WLSValues(WLS{poly_order, bspline_order, dim}(), N, ∇N, w, Ref(M⁻¹))
end

function ShapeValues(::Type{T}, F::WLS{poly_order, bspline_order, dim}) where {poly_order, bspline_order, dim, T}
    p = polynomial(F)
    L = length(p(zero(Vec{dim, T})))
    WLSValues{poly_order, bspline_order, dim, T, L, L^2}()
end

function reinit!(it::WLSValues{<: Any, <: Any, dim}, grid::Grid{dim}, x::Vec{dim}, indices::AbstractArray = CartesianIndices(grid)) where {dim}
    @boundscheck checkbounds(grid, indices)
    F = weight_function(it)
    resize!(it.N, length(indices))
    resize!(it.∇N, length(indices))
    resize!(it.w, length(indices))
    @inbounds @simd for i in eachindex(indices)
        I = indices[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) ./ gridsteps(grid)
        it.w[i] = F(ξ)
    end
    P = polynomial(it)
    M = zero(it.M⁻¹[])
    @inbounds @simd for i in eachindex(indices)
        I = indices[i]
        xᵢ = grid[I]
        p = P(xᵢ - x)
        M += it.w[i] * p ⊗ p
    end
    it.M⁻¹[] = inv(M)
    p₀ = P(x - x)
    ∇p₀ = P'(x - x)
    @inbounds @simd for i in eachindex(indices)
        I = indices[i]
        xᵢ = grid[I]
        q = it.M⁻¹[] ⋅ P(xᵢ - x)
        wq = it.w[i] * q
        it.N[i] = wq ⋅ p₀
        it.∇N[i] = wq ⋅ ∇p₀
    end
end


struct WLSValue{dim, T, L, M}
    N::T
    ∇N::Vec{dim, T}
    w::T
    M⁻¹::Mat{L, L, T, M}
end

@inline function Base.getindex(it::WLSValues, i::Int)
    @_propagate_inbounds_meta
    WLSValue(it.N[i], it.∇N[i], it.w[i], it.M⁻¹[])
end
