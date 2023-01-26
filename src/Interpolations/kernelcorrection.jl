struct KernelCorrection{K <: Kernel} <: Interpolation
end
@pure KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

@pure get_kernel(::KernelCorrection{K}) where {K} = K()

@inline function nodeindices(x::KernelCorrection, grid::Grid, pt)
    nodeindices(get_kernel(x), grid, pt)
end


mutable struct KernelCorrectionValue{K, dim, T} <: MPValue{dim, T}
    F::KernelCorrection{K}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
    nodeindices::CartesianIndices{dim, NTuple{dim, UnitRange{Int}}} # necessary in MPValue
end

function MPValue{dim, T}(F::KernelCorrection) where {dim, T}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    nodeindices = CartesianIndices(nfill(1:0, Val(dim)))
    KernelCorrectionValue(F, N, ∇N, nodeindices)
end

get_kernel(mp::KernelCorrectionValue) = get_kernel(mp.F)

function update_kernels!(mp::KernelCorrectionValue{<: Any, dim, T}, grid::Grid{dim}, sppat::AbstractArray{Bool, dim}, pt) where {dim, T}
    n = num_nodes(mp)

    # reset
    resize_fillzero!(mp.N, n)
    resize_fillzero!(mp.∇N, n)

    # update
    F = get_kernel(mp)
    xp = getx(pt)
    if n == maxnum_nodes(F, Val(dim)) && all(@inbounds view(sppat, mp.nodeindices)) # all active
        wᵢ, ∇wᵢ = values_gradients(F, grid, pt)
        mp.N .= wᵢ
        mp.∇N .= ∇wᵢ
    else
        M = zero(Mat{dim+1, dim+1, T})
        @inbounds for (j, i) in enumerate(mp.nodeindices)
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
        @inbounds for (j, i) in enumerate(mp.nodeindices)
            xi = grid[i]
            w = mp.N[j]
            mp.N[j] = (C1 + C2 ⋅ (xi - xp)) * w
            mp.∇N[j] = (C2 + C3 ⋅ (xi - xp)) * w
        end
    end

    mp
end
