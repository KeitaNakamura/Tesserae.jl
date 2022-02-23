struct WLS{B <: AbstractBasis, K <: Kernel} <: Interpolation
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

@pure WLS{B}(w::Kernel) where {B} = WLS{B, typeof(w)}()

@pure getbasisfunction(::WLS{B}) where {B} = B()
@pure getkernelfunction(::WLS{B, W}) where {B, W} = W()

getsupportlength(wls::WLS, args...) = getsupportlength(getkernelfunction(wls), args...)


mutable struct WLSValues{B, K, dim, T, nnodes, L, L²} <: MPValues{dim, T}
    F::WLS{B, K}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    w::MVector{nnodes, T}
    gridindices::MVector{nnodes, Index{dim}}
    M⁻¹::Mat{L, L, T, L²}
    x::Vec{dim, T}
    len::Int
end

getbasisfunction(x::WLSValues) = getbasisfunction(x.F)
getkernelfunction(x::WLSValues) = getkernelfunction(x.F)

function WLSValues{B, K, dim, T, nnodes, L, L²}() where {B, K, dim, T, nnodes, L, L²}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    w = MVector{nnodes, T}(undef)
    M⁻¹ = zero(Mat{L, L, T, L²})
    gridindices = MVector{nnodes, Index{dim}}(undef)
    x = zero(Vec{dim, T})
    WLSValues(WLS{B, K}(), N, ∇N, w, gridindices, M⁻¹, x, 0)
end

function MPValues{dim, T}(F::WLS{B, K}) where {B, K, dim, T}
    L = length(value(getbasisfunction(F), zero(Vec{dim, T})))
    n = nnodes(getkernelfunction(F), Val(dim))
    WLSValues{B, K, dim, T, n, L, L^2}()
end

function _update!(mpvalues::WLSValues, F, grid::Grid, x::Vec, spat::AbstractArray{Bool}, inds)
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    fillzero!(mpvalues.w)
    P = getbasisfunction(mpvalues)
    M = zero(mpvalues.M⁻¹)
    mpvalues.x = x
    update_gridindices!(mpvalues, inds, spat)
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

function _update!(mpvalues::WLSValues{PolynomialBasis{1}, <: BSpline, dim, T}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}, inds) where {dim, T}
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    fillzero!(mpvalues.w)
    P = getbasisfunction(mpvalues)
    mpvalues.x = x

    dx⁻¹ = gridsteps_inv(grid)
    iscompleted = update_gridindices!(mpvalues, inds, spat)
    if iscompleted
        # fast version
        D = zero(Vec{dim, T}) # diagonal entries
        wᵢ = values(getkernelfunction(mpvalues), x .* dx⁻¹)
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
    F = getkernelfunction(mpvalues)
    _update!(mpvalues, ξ -> value(F, ξ), grid, x, spat, neighboring_nodes(grid, x, getsupportlength(F)))
end

function update!(mpvalues::WLSValues{<: Any, GIMP}, grid::Grid, x::Vec, r::Vec, spat::AbstractArray{Bool})
    F = getkernelfunction(mpvalues)
    dx⁻¹ = gridsteps_inv(grid)
    rdx⁻¹ = r.*dx⁻¹
    _update!(mpvalues, ξ -> value(F, ξ, rdx⁻¹), grid, x, spat, neighboring_nodes(grid, x, getsupportlength(F, rdx⁻¹)))
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
