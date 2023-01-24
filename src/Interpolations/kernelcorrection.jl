struct KernelCorrection{K <: Kernel} <: Interpolation
end
@pure KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

@pure get_kernel(::KernelCorrection{K}) where {K} = K()

@inline function nodeindices(x::KernelCorrection, grid::Grid, pt)
    nodeindices(get_kernel(x), grid, pt)
end


mutable struct KernelCorrectionValue{K, dim, T, L} <: MPValue{dim, T}
    F::KernelCorrection{K}
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    # necessary in MPValue
    nodeindices::MVector{L, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

function MPValue{dim, T}(F::KernelCorrection) where {dim, T}
    L = num_nodes(get_kernel(F), Val(dim))
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    nodeindices = MVector{L, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    KernelCorrectionValue(F, N, ∇N, nodeindices, xp, 0)
end

get_kernel(mp::KernelCorrectionValue) = get_kernel(mp.F)

function update_kernels!(mp::KernelCorrectionValue{<: Any, dim, T, L}, grid::Grid{dim}, pt) where {dim, T, L}
    # reset
    fillzero!(mp.N)
    fillzero!(mp.∇N)

    # update
    F = get_kernel(mp)
    xp = getx(pt)
    if num_nodes(mp) == L # all active
        wᵢ, ∇wᵢ = values_gradients(F, grid, pt)
        mp.N .= wᵢ
        mp.∇N .= ∇wᵢ
    else
        M = zero(Mat{dim+1, dim+1, T})
        @inbounds @simd for j in 1:num_nodes(mp)
            i = mp.nodeindices[j]
            xi = grid[i]
            w = value(F, grid, i, pt)
            P = [1; xi - xp]
            M += w * P ⊗ P
            mp.N[j] = w
        end
        Minv = inv(M)
        C1 = Minv[1,1]
        C2 = @Tensor Minv[2:end,1]
        C3 = @Tensor Minv[2:end,2:end]
        @inbounds @simd for j in 1:num_nodes(mp)
            i = mp.nodeindices[j]
            xi = grid[i]
            w = mp.N[j]
            mp.N[j] = (C1 + C2 ⋅ (xi - xp)) * w
            mp.∇N[j] = (C2 + C3 ⋅ (xi - xp)) * w
        end
    end

    mp
end
