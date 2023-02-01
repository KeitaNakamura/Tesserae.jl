struct KernelCorrection{K <: Kernel} <: Interpolation
end
@pure KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()
@pure get_kernel(::KernelCorrection{K}) where {K} = K()

struct KernelCorrectionValue{dim, T, K} <: MPValue{dim, T, KernelCorrection{K}}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function MPValue{dim, T}(::KernelCorrection{K}) where {dim, T, K}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    KernelCorrectionValue{dim, T, K}(N, ∇N)
end

function update_kernels!(mp::KernelCorrectionValue{dim, T}, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::AbstractArray, pt) where {dim, T}
    n = length(nodeinds)

    # reset
    resize!(mp.N, n)
    resize!(mp.∇N, n)

    # update
    F = get_kernel(mp)
    if n == maxnum_nodes(F, Val(dim)) && all(@inbounds view(sppat, nodeinds)) # all active
        wᵢ, ∇wᵢ = values_gradients(F, grid, pt)
        mp.N .= wᵢ
        mp.∇N .= ∇wᵢ
    else
        xp = getx(pt)
        M = zero(Mat{dim+1, dim+1, T})
        @inbounds for (j, i) in enumerate(nodeinds)
            xi = grid[i]
            w = value(F, grid, i, pt) * sppat[i]
            P = [1; xi - xp]
            M += w * P ⊗ P
            mp.N[j] = w
        end
        Minv = inv(M)
        C1 = Minv[1,1]
        C2 = @Tensor Minv[2:end,1]
        C3 = @Tensor Minv[2:end,2:end]
        @inbounds for (j, i) in enumerate(nodeinds)
            xi = grid[i]
            w = mp.N[j]
            mp.N[j] = (C1 + C2 ⋅ (xi - xp)) * w
            mp.∇N[j] = (C2 + C3 ⋅ (xi - xp)) * w
        end
    end

    mp
end
