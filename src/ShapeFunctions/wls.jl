abstract type AbstractPolynomial end

struct Polynomial{order} <: AbstractPolynomial
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


struct Bilinear <: AbstractPolynomial end

value(poly::Bilinear, x::Vec{2, T}) where {T} = @inbounds Vec(one(T), x[1], x[2], x[1]*x[2])
function Tensorial.gradient(poly::Bilinear, x::Vec{2, T}) where {T}
    z = zero(T)
    o = one(T)
    @Mat [z    z
          o    z
          z    o
          x[2] x[1]]
end

# for ∇ operation
struct PolynomialGradient{P}
    parent::P
end
Base.adjoint(x::AbstractPolynomial) = PolynomialGradient(x)

# function like methods
(p::AbstractPolynomial)(x) = value(p, x)
(p::PolynomialGradient)(x) = gradient(p.parent, x)


struct WLS{P <: AbstractPolynomial, W <: ShapeFunction} <: ShapeFunction
    poly::P
    weight::W
end

const LinearWLS = WLS{Polynomial{1}}
const BilinearWLS = WLS{Bilinear}

WLS{P, W}() where {P, W} = WLS(P(), W())
WLS{P}(weight::ShapeFunction) where {P} = WLS(P(), weight)

polynomial(wls::WLS) = wls.poly
weight_function(wls::WLS) = wls.weight

support_length(wls::WLS, args...) = support_length(weight_function(wls), args...)
active_length(::WLS, args...) = 1.0 # for sparsity pattern


struct WLSValues{P, W, dim, T, L, M, O} <: ShapeValues{dim, T}
    F::WLS{P, W}
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    w::MVector{L, T}
    M⁻¹::Base.RefValue{Mat{M, M, T, O}}
    x::Base.RefValue{Vec{dim, T}}
    inds::MVector{L, Index{dim}}
    len::Base.RefValue{Int}
end

polynomial(it::WLSValues) = polynomial(it.F)
weight_function(it::WLSValues) = weight_function(it.F)

function WLSValues{P, W, dim, T, L, M, O}() where {P, W, dim, T, L, M, O}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    w = MVector{L, T}(undef)
    M⁻¹ = zero(Mat{M, M, T, O})
    inds = MVector{L, Index{dim}}(undef)
    x = Ref(zero(Vec{dim, T}))
    WLSValues(WLS{P, W}(), N, ∇N, w, Ref(M⁻¹), x, inds, Ref(0))
end

function ShapeValues{dim, T}(F::WLS{P, W}) where {P, W, dim, T}
    p = polynomial(F)
    M = length(p(zero(Vec{dim, T})))
    L = nnodes(weight_function(F), Val(dim))
    WLSValues{P, W, dim, T, L, M, M^2}()
end

function _update!(it::WLSValues{<: Any, <: Any, dim}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    it.N .= zero(it.N)
    it.∇N .= zero(it.∇N)
    it.w .= zero(it.w)
    P = polynomial(it)
    M = zero(it.M⁻¹[])
    it.x[] = x
    update_gridindices!(it, grid, x, spat)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) ./ gridsteps(grid)
        w = F(ξ)
        p = P(xᵢ - x)
        M += w * p ⊗ p
        it.w[i] = w
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

function _update!(it::WLSValues{Polynomial{1}, <: Any, dim}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    it.N .= zero(it.N)
    it.∇N .= zero(it.∇N)
    it.w .= zero(it.w)
    P = polynomial(it)
    M = zero(it.M⁻¹[])
    it.x[] = x
    update_gridindices!(it, grid, x, spat)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) ./ gridsteps(grid)
        w = F(ξ)
        p = P(xᵢ - x)
        M += w * p ⊗ p
        it.w[i] = w
    end
    it.M⁻¹[] = inv(M)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        q = it.M⁻¹[] ⋅ P(xᵢ - x)
        wq = it.w[i] * q
        it.N[i] = @Tensor wq[1]
        it.∇N[i] = @Tensor wq[2:end]
    end
    it
end

function update!(it::WLSValues{<: Any, <: Any, dim}, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    F = weight_function(it)
    _update!(it, ξ -> value(F, ξ), grid, x, spat)
end

function update!(it::WLSValues{<: Any, GIMP, dim}, grid::Grid{dim}, x::Vec{dim}, r::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    F = weight_function(it)
    _update!(it, ξ -> value(F, ξ, r./gridsteps(grid)), grid, x, spat)
end


struct WLSValue{dim, T, L, M}
    N::T
    ∇N::Vec{dim, T}
    w::T
    M⁻¹::Mat{L, L, T, M}
    x::Vec{dim, T}
    index::Index{dim}
end

@inline function Base.getindex(it::WLSValues, i::Int)
    @_propagate_inbounds_meta
    WLSValue(it.N[i], it.∇N[i], it.w[i], it.M⁻¹[], it.x[], it.inds[i])
end
