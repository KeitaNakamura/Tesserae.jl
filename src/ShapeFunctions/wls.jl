struct WLS{Basis <: AbstractBasis, Weight <: ShapeFunction} <: ShapeFunction
    basis::Basis
    weight::Weight
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

WLS{Basis, Weight}() where {Basis, Weight} = WLS(Basis(), Weight())
WLS{Basis}(weight::ShapeFunction) where {Basis} = WLS(Basis(), weight)

basis_function(wls::WLS) = wls.basis
weight_function(wls::WLS) = wls.weight

support_length(wls::WLS, args...) = support_length(weight_function(wls), args...)
active_length(::WLS, args...) = 1.0 # for sparsity pattern


struct WLSValues{Basis, Weight, dim, T, nnodes, L, L²} <: ShapeValues{dim, T}
    F::WLS{Basis, Weight}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    w::MVector{nnodes, T}
    M⁻¹::Base.RefValue{Mat{L, L, T, L²}}
    x::Base.RefValue{Vec{dim, T}}
    inds::MVector{nnodes, Index{dim}}
    len::Base.RefValue{Int}
end

basis_function(it::WLSValues) = basis_function(it.F)
weight_function(it::WLSValues) = weight_function(it.F)

function WLSValues{Basis, Weight, dim, T, nnodes, L, L²}() where {Basis, Weight, dim, T, nnodes, L, L²}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    w = MVector{nnodes, T}(undef)
    M⁻¹ = zero(Mat{L, L, T, L²})
    inds = MVector{nnodes, Index{dim}}(undef)
    x = Ref(zero(Vec{dim, T}))
    WLSValues(WLS{Basis, Weight}(), N, ∇N, w, Ref(M⁻¹), x, inds, Ref(0))
end

function ShapeValues{dim, T}(F::WLS{Basis, Weight}) where {Basis, Weight, dim, T}
    L = length(value(basis_function(F), zero(Vec{dim, T})))
    n = nnodes(weight_function(F), Val(dim))
    WLSValues{Basis, Weight, dim, T, n, L, L^2}()
end

function _update!(it::WLSValues{<: Any, <: Any, dim}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    it.N .= zero(it.N)
    it.∇N .= zero(it.∇N)
    it.w .= zero(it.w)
    P = basis_function(it)
    M = zero(it.M⁻¹[])
    it.x[] = x
    update_gridindices!(it, grid, x, spat)
    dx⁻¹ = 1 ./ gridsteps(grid)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) .* dx⁻¹
        w = F(ξ)
        p = value(P, xᵢ - x)
        M += w * p ⊗ p
        it.w[i] = w
    end
    it.M⁻¹[] = inv(M)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        q = it.M⁻¹[] ⋅ value(P, xᵢ - x)
        wq = it.w[i] * q
        it.N[i] = wq ⋅ value(P, x - x)
        it.∇N[i] = wq ⋅ gradient(P, x - x)
    end
    it
end

function _update!(it::WLSValues{PolynomialBasis{1}, <: Any, dim}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    it.N .= zero(it.N)
    it.∇N .= zero(it.∇N)
    it.w .= zero(it.w)
    P = basis_function(it)
    M = zero(it.M⁻¹[])
    it.x[] = x
    update_gridindices!(it, grid, x, spat)
    dx⁻¹ = 1 ./ gridsteps(grid)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) .* dx⁻¹
        w = F(ξ)
        p = value(P, xᵢ - x)
        M += w * p ⊗ p
        it.w[i] = w
    end
    it.M⁻¹[] = inv(M)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        q = it.M⁻¹[] ⋅ value(P, xᵢ - x)
        wq = it.w[i] * q
        it.N[i] = wq[1]
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
    dx⁻¹ = 1 ./ gridsteps(grid)
    _update!(it, ξ -> value(F, ξ, r.*dx⁻¹), grid, x, spat)
end


struct WLSValue{dim, T, L, L²}
    N::T
    ∇N::Vec{dim, T}
    w::T
    M⁻¹::Mat{L, L, T, L²}
    x::Vec{dim, T}
    index::Index{dim}
end

@inline function Base.getindex(it::WLSValues, i::Int)
    @_propagate_inbounds_meta
    WLSValue(it.N[i], it.∇N[i], it.w[i], it.M⁻¹[], it.x[], it.inds[i])
end
