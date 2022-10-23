struct WLS{B <: AbstractBasis, K <: Kernel} <: Interpolation
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

@pure WLS{B}(w::Kernel) where {B} = WLS{B, typeof(w)}()

@pure get_basis(::WLS{B}) where {B} = B()
@pure get_kernel(::WLS{B, W}) where {B, W} = W()

@inline function nodeindices(wls::WLS, grid::Grid, pt)
    nodeindices(get_kernel(wls), grid, pt)
end


mutable struct WLSValue{W <: WLS, dim, T, L, Minv_T <: Mat{<: Any, <: Any, T}} <: MPValue{dim, T}
    F::W
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    w::MVector{L, T}
    Minv::Minv_T
    # necessary in MPValue
    xp::Vec{dim, T}
    nodeindices::MVector{L, Index{dim}}
    len::Int
end

function MPValue{dim, T}(F::WLS) where {dim, T}
    L = num_nodes(get_kernel(F), Val(dim))
    n = length(value(get_basis(F), zero(Vec{dim, T})))
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    w = MVector{L, T}(undef)
    Minv = zero(Mat{n,n,T,n^2})
    xp = zero(Vec{dim, T})
    nodeindices = MVector{L, Index{dim}}(undef)
    WLSValue(F, N, ∇N, w, Minv, xp, nodeindices, 0)
end

get_kernel(mp::WLSValue) = get_kernel(mp.F)
get_basis(mp::WLSValue) = get_basis(mp.F)

@inline function mpvalue(mp::WLSValue, i::Int)
    @boundscheck @assert 1 ≤ i ≤ num_nodes(mp)
    (; N=mp.N[i], ∇N=mp.∇N[i], w=mp.w[i], Minv=mp.Minv, xp=mp.xp)
end

# general version
function update_kernels!(mp::WLSValue, grid::Grid, pt)
    # reset
    fillzero!(mp.N)
    fillzero!(mp.∇N)
    fillzero!(mp.w)

    # update
    F = get_kernel(mp)
    P = get_basis(mp)
    M = zero(mp.Minv)
    xp = getx(pt)
    @inbounds @simd for i in 1:num_nodes(mp)
        I = nodeindex(mp, i)
        xi = grid[I]
        w = value(F, grid, I, pt)
        p = value(P, xi - xp)
        M += w * p ⊗ p
        mp.w[i] = w
    end
    Minv = inv(M)
    @inbounds @simd for i in 1:num_nodes(mp)
        I = nodeindex(mp, i)
        xi = grid[I]
        q = Minv ⋅ value(P, xi - xp)
        wq = mp.w[i] * q
        mp.N[i] = wq ⋅ value(P, xp - xp)
        mp.∇N[i] = wq ⋅ gradient(P, xp - xp)
    end
    mp.Minv = Minv

    mp
end

# fast version for `LinearWLS(BSpline{order}())`
function update_kernels!(mp::WLSValue{<: LinearWLS{<: BSpline}, dim, T, L}, grid::Grid{dim}, pt) where {dim, T, L}
    # reset
    fillzero!(mp.N)
    fillzero!(mp.∇N)
    fillzero!(mp.w)

    # update
    F = get_kernel(mp)
    P = get_basis(mp)
    xp = getx(pt)
    if num_nodes(mp) == L # all activate
        # fast version
        D = zero(Vec{dim, T}) # diagonal entries
        wᵢ = values(F, grid, xp)
        @inbounds @simd for i in 1:num_nodes(mp)
            I = nodeindex(mp, i)
            xi = grid[I]
            w = wᵢ[i]
            D += w * (xi - xp) .* (xi - xp)
            mp.w[i] = w
            mp.∇N[i] = w * (xi - xp)
        end
        D⁻¹ = inv.(D)
        @inbounds @simd for i in 1:num_nodes(mp)
            mp.N[i] = wᵢ[i]
            mp.∇N[i] = mp.∇N[i] .* D⁻¹
        end
        mp.Minv = diagm(vcat(1, D⁻¹))
    else
        M = zero(mp.Minv)
        @inbounds @simd for i in 1:num_nodes(mp)
            I = nodeindex(mp, i)
            xi = grid[I]
            w = value(F, grid, I, xp)
            p = value(P, xi - xp)
            M += w * p ⊗ p
            mp.w[i] = w
        end
        Minv = inv(M)
        @inbounds @simd for i in 1:num_nodes(mp)
            I = nodeindex(mp, i)
            xi = grid[I]
            q = Minv ⋅ value(P, xi - xp)
            wq = mp.w[i] * q
            mp.N[i] = wq[1]
            mp.∇N[i] = @Tensor wq[2:end]
        end
        mp.Minv = Minv
    end

    mp
end
