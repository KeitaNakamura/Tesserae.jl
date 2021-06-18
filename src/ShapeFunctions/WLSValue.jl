struct WLSValue{order, weight_order, dim, T, L, M} <: ShapeValue{dim, T}
    F::WLS{order, weight_order, dim}
    N::Vector{T}
    dN::Vector{Vec{dim, T}}
    w::Vector{T}
    M⁻¹::Base.RefValue{Mat{L, L, T, M}}
end

polynomial(it::WLSValue) = polynomial(it.F)
weight_function(it::WLSValue) = weight_function(it.F)

weight_value(it::WLSValue) = Collection{1}(it.w)
moment_matrix_inverse(it::WLSValue) = it.M⁻¹[]

function construct(::Type{T}, F::WLS{order, weight_order, dim}) where {order, weight_order, dim, T}
    p = polynomial(F)
    L = length(p(zero(Vec{dim, T})))
    N = Vector{T}(undef, 0)
    dN = Vector{Vec{dim, T}}(undef, 0)
    w = Vector{T}(undef, 0)
    M⁻¹ = zero(Mat{L, L, T})
    WLSValue(F, N, dN, w, Ref(M⁻¹))
end

function reinit!(it::WLSValue{<: Any, <: Any, dim}, grid::AbstractGrid{dim}, indices::AbstractArray, x::Vec{dim}) where {dim}
    @boundscheck checkbounds(grid, indices)
    F = weight_function(it)
    resize!(it.N, length(indices))
    resize!(it.dN, length(indices))
    resize!(it.w, length(indices))
    @inbounds for (j, I) in enumerate(view(CartesianIndices(grid), indices))
        xᵢ = grid[I]
        ξ = (x - xᵢ) ./ gridsteps(grid)
        it.w[j] = F(ξ)
    end
    P = polynomial(it)
    M = zero(it.M⁻¹[])
    @inbounds for (j, I) in enumerate(view(CartesianIndices(grid), indices))
        xᵢ = grid[I]
        p = P(xᵢ - x)
        M += it.w[j] * p ⊗ p
    end
    it.M⁻¹[] = inv(M)
    p₀ = P(x - x)
    ∇p₀ = ∇(P)(x - x)
    @inbounds for (j, I) in enumerate(view(CartesianIndices(grid), indices))
        xᵢ = grid[I]
        q = it.M⁻¹[] ⋅ P(xᵢ - x)
        wq = it.w[j] * q
        it.N[j] = wq ⋅ p₀
        it.dN[j] = wq ⋅ ∇p₀
    end
end
