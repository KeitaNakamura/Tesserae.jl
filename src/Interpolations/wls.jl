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
    nodeindices::MVector{L, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

function MPValue{dim, T}(F::WLS) where {dim, T}
    L = num_nodes(get_kernel(F), Val(dim))
    n = length(value(get_basis(F), zero(Vec{dim, T})))
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    w = MVector{L, T}(undef)
    Minv = zero(Mat{n,n,T,n^2})
    nodeindices = MVector{L, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    WLSValue(F, N, ∇N, w, Minv, nodeindices, xp, 0)
end

get_kernel(mp::WLSValue) = get_kernel(mp.F)
get_basis(mp::WLSValue) = get_basis(mp.F)

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
    @inbounds @simd for j in 1:num_nodes(mp)
        i = mp.nodeindices[j]
        xi = grid[i]
        w = value(F, grid, i, pt)
        p = value(P, xi - xp)
        M += w * p ⊗ p
        mp.w[j] = w
    end
    Minv = inv(M)
    @inbounds @simd for j in 1:num_nodes(mp)
        i = mp.nodeindices[j]
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
        wᵢ = first(values_gradients(F, grid, xp))
        @inbounds @simd for j in 1:num_nodes(mp)
            i = mp.nodeindices[j]
            xi = grid[i]
            w = wᵢ[j]
            D += w * (xi - xp) .* (xi - xp)
            mp.w[j] = w
            mp.∇N[j] = w * (xi - xp)
        end
        D⁻¹ = inv.(D)
        @inbounds @simd for j in 1:num_nodes(mp)
            mp.N[j] = wᵢ[j]
            mp.∇N[j] = mp.∇N[j] .* D⁻¹
        end
        mp.Minv = diagm(vcat(1, D⁻¹))
    else
        M = zero(mp.Minv)
        @inbounds @simd for j in 1:num_nodes(mp)
            i = mp.nodeindices[j]
            xi = grid[i]
            w = value(F, grid, i, xp)
            p = value(P, xi - xp)
            M += w * p ⊗ p
            mp.w[j] = w
        end
        Minv = inv(M)
        @inbounds @simd for j in 1:num_nodes(mp)
            i = mp.nodeindices[j]
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
