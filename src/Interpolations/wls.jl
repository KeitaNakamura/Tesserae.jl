struct WLS{Basis <: AbstractBasis, Weight <: Kernel} <: Interpolation
    basis::Basis
    weight::Weight
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

WLS{Basis, Weight}() where {Basis, Weight} = WLS(Basis(), Weight())
WLS{Basis}(weight::Kernel) where {Basis} = WLS(Basis(), weight)

basis_function(wls::WLS) = wls.basis
weight_function(wls::WLS) = wls.weight

support_length(wls::WLS, args...) = support_length(weight_function(wls), args...)
active_length(::WLS, args...) = 1.0 # for sparsity pattern


mutable struct WLSValues{Basis, Weight, dim, T, nnodes, L, L²} <: MPValues{dim, T}
    F::WLS{Basis, Weight}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    w::MVector{nnodes, T}
    gridindices::MVector{nnodes, Index{dim}}
    M⁻¹::Mat{L, L, T, L²}
    x::Vec{dim, T}
    len::Int
end

basis_function(x::WLSValues) = basis_function(x.F)
weight_function(x::WLSValues) = weight_function(x.F)

function WLSValues{Basis, Weight, dim, T, nnodes, L, L²}() where {Basis, Weight, dim, T, nnodes, L, L²}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    w = MVector{nnodes, T}(undef)
    M⁻¹ = zero(Mat{L, L, T, L²})
    gridindices = MVector{nnodes, Index{dim}}(undef)
    x = zero(Vec{dim, T})
    WLSValues(WLS{Basis, Weight}(), N, ∇N, w, gridindices, M⁻¹, x, 0)
end

function MPValues{dim, T}(F::WLS{Basis, Weight}) where {Basis, Weight, dim, T}
    L = length(value(basis_function(F), zero(Vec{dim, T})))
    n = nnodes(weight_function(F), Val(dim))
    WLSValues{Basis, Weight, dim, T, n, L, L^2}()
end

function _update!(mpvalues::WLSValues, F, grid::Grid, x::Vec, spat::AbstractArray{Bool})
    mpvalues.N .= elzero(mpvalues.N)
    mpvalues.∇N .= elzero(mpvalues.∇N)
    mpvalues.w .= elzero(mpvalues.w)
    P = basis_function(mpvalues)
    M = zero(mpvalues.M⁻¹)
    mpvalues.x = x
    update_gridindices!(mpvalues, grid, x, spat)
    dx⁻¹ = gridsteps_inv(grid)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) .* dx⁻¹
        w = F(ξ)
        p = value(P, xᵢ - x)
        M += w * p ⊗ p
        mpvalues.w[i] = w
    end
    mpvalues.M⁻¹ = inv(M)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        q = mpvalues.M⁻¹ ⋅ value(P, xᵢ - x)
        wq = mpvalues.w[i] * q
        mpvalues.N[i] = wq ⋅ value(P, x - x)
        mpvalues.∇N[i] = wq ⋅ gradient(P, x - x)
    end
    mpvalues
end

function _update!(mpvalues::WLSValues{PolynomialBasis{1}, <: BSpline, dim, T}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim, T}
    mpvalues.N .= elzero(mpvalues.N)
    mpvalues.∇N .= elzero(mpvalues.∇N)
    mpvalues.w .= elzero(mpvalues.w)
    P = basis_function(mpvalues)
    mpvalues.x = x

    iscompleted = update_gridindices!(mpvalues, grid, x, spat)
    dx⁻¹ = gridsteps_inv(grid)
    if iscompleted
        # fast version
        D = zero(Vec{dim, T}) # diagonal entries
        wᵢ = values(weight_function(mpvalues), x .* dx⁻¹)
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            w = wᵢ[i]
            D += w * (xᵢ - x) .* (xᵢ - x)
            mpvalues.w[i] = w
            mpvalues.∇N[i] = w * (xᵢ - x) # for fast computation
        end
        D⁻¹ = inv.(D)
        @inbounds @simd for i in 1:length(mpvalues)
            mpvalues.N[i] = wᵢ[i]
            mpvalues.∇N[i] = mpvalues.∇N[i] .* D⁻¹
        end
        M⁻¹ = diagm(vcat(1, D⁻¹))
    else
        M = zero(mpvalues.M⁻¹)
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            ξ = (x - xᵢ) .* dx⁻¹
            w = F(ξ)
            p = value(P, xᵢ - x)
            M += w * p ⊗ p
            mpvalues.w[i] = w
        end
        M⁻¹ = inv(M)
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            q = M⁻¹ ⋅ value(P, xᵢ - x)
            wq = mpvalues.w[i] * q
            mpvalues.N[i] = wq[1]
            mpvalues.∇N[i] = @Tensor wq[2:end]
        end
    end
    mpvalues.M⁻¹ = M⁻¹

    mpvalues
end

function update!(mpvalues::WLSValues, grid::Grid, x::Vec, spat::AbstractArray{Bool})
    F = weight_function(mpvalues)
    _update!(mpvalues, ξ -> value(F, ξ), grid, x, spat)
end

function update!(mpvalues::WLSValues{<: Any, GIMP}, grid::Grid, x::Vec, r::Vec, spat::AbstractArray{Bool})
    F = weight_function(mpvalues)
    dx⁻¹ = gridsteps_inv(grid)
    _update!(mpvalues, ξ -> value(F, ξ, r.*dx⁻¹), grid, x, spat)
end


struct WLSValue{dim, T, L, L²} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    w::T
    I::Index{dim}
    M⁻¹::Mat{L, L, T, L²}
    x::Vec{dim, T}
end

@inline function Base.getindex(mpvalues::WLSValues, i::Int)
    @_propagate_inbounds_meta
    WLSValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.w[i], mpvalues.gridindices[i], mpvalues.M⁻¹, mpvalues.x)
end
