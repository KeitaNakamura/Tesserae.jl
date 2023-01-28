struct WLS{B <: AbstractBasis, K <: Kernel} <: Interpolation
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

@pure WLS{B}(w::Kernel) where {B} = WLS{B, typeof(w)}()

@pure get_basis(::WLS{B}) where {B} = B()
@pure get_kernel(::WLS{B, W}) where {B, W} = W()

mutable struct WLSValue{dim, T, B, K, L, L²} <: MPValue{dim, T, WLS{B, K}}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
    w::Vector{T}
    Minv::Mat{L, L, T, L²}
end

function MPValue{dim, T}(F::WLS{B, K}) where {dim, T, B, K}
    L = length(value(get_basis(F), zero(Vec{dim, T})))
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    w = Vector{T}(undef, 0)
    Minv = zero(Mat{L, L, T})
    WLSValue{dim, T, B, K, L, L^2}(N, ∇N, w, Minv)
end

get_basis(x::WLSValue) = get_basis(get_interp(x))

# general version
function update_kernels!(mp::WLSValue, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::AbstractArray, pt)
    n = length(nodeinds)

    # reset
    resize_fillzero!(mp.N, n)
    resize_fillzero!(mp.∇N, n)
    resize_fillzero!(mp.w, n)

    # update
    F = get_kernel(mp)
    P = get_basis(mp)
    M = zero(mp.Minv)
    xp = getx(pt)
    @inbounds for (j, i) in enumerate(nodeinds)
        xi = grid[i]
        w = value(F, grid, i, pt) * sppat[i]
        p = value(P, xi - xp)
        M += w * p ⊗ p
        mp.w[j] = w
    end
    Minv = inv(M)
    @inbounds for (j, i) in enumerate(nodeinds)
        xi = grid[i]
        q = Minv ⋅ value(P, xi - xp)
        wq = mp.w[j] * q
        mp.N[j] = wq ⋅ value(P, xp - xp)
        mp.∇N[j] = wq ⋅ gradient(P, xp - xp)
    end
    mp.Minv = Minv

    mp
end

# fast version for `LinearWLS(BSpline{order}())`
function update_kernels!(mp::WLSValue{dim, T, PolynomialBasis{1}, <: BSpline}, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::AbstractArray, pt) where {dim, T}
    n = length(nodeinds)

    # reset
    resize_fillzero!(mp.N, n)
    resize_fillzero!(mp.∇N, n)
    resize_fillzero!(mp.w, n)

    # update
    F = get_kernel(mp)
    P = get_basis(mp)
    xp = getx(pt)
    if n == maxnum_nodes(F, Val(dim)) && all(@inbounds view(sppat, nodeinds)) # all activate
        # fast version
        D = zero(Vec{dim, T}) # diagonal entries
        wᵢ = first(values_gradients(F, grid, xp))
        @inbounds for (j, i) in enumerate(nodeinds)
            xi = grid[i]
            w = wᵢ[j] * sppat[i]
            D += w * (xi - xp) .* (xi - xp)
            mp.w[j] = w
            mp.∇N[j] = w * (xi - xp)
        end
        D⁻¹ = inv.(D)
        @inbounds for j in 1:n
            mp.N[j] = wᵢ[j]
            mp.∇N[j] = mp.∇N[j] .* D⁻¹
        end
        mp.Minv = diagm(vcat(1, D⁻¹))
    else
        M = zero(mp.Minv)
        @inbounds for (j, i) in enumerate(nodeinds)
            xi = grid[i]
            w = value(F, grid, i, xp) * sppat[i]
            p = value(P, xi - xp)
            M += w * p ⊗ p
            mp.w[j] = w
        end
        Minv = inv(M)
        @inbounds for (j, i) in enumerate(nodeinds)
            xi = grid[i]
            q = Minv ⋅ value(P, xi - xp)
            wq = mp.w[j] * q
            mp.N[j] = wq[1]
            mp.∇N[j] = @Tensor wq[2:end]
        end
        mp.Minv = Minv
    end

    mp
end
