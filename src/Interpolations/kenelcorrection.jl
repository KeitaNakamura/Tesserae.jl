struct KernelCorrection{K <: Kernel} <: Interpolation
end
@pure KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

@pure get_kernel(::KernelCorrection{K}) where {K} = K()

@inline function gridindices(x::KernelCorrection, grid::Grid, pt)
    gridindices(get_kernel(x), grid, pt)
end


struct KernelCorrectionValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    xp::Vec{dim, T}
end

mutable struct KernelCorrectionValues{K, dim, T, nnodes} <: MPValues{dim, T, KernelCorrectionValue{dim, T}}
    F::KernelCorrection{K}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    gridindices::MVector{nnodes, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

# constructors
function KernelCorrectionValues{K, dim, T, nnodes}() where {K, dim, T, nnodes}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    gridindices = MVector{nnodes, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    KernelCorrectionValues(KernelCorrection(K()), N, ∇N, gridindices, xp, 0)
end
function MPValues{dim, T}(c::KernelCorrection{K}) where {dim, T, K}
    L = num_nodes(K(), Val(dim))
    KernelCorrectionValues{K, dim, T, L}()
end

get_kernel(c::KernelCorrectionValues) = get_kernel(c.F)

function update!(mpvalues::KernelCorrectionValues{<: Any, dim, T}, grid::Grid{<: Any, dim}, pt, spat::AbstractArray{Bool, dim}) where {dim, T}
    # reset
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)

    F = get_kernel(mpvalues)
    xp = getx(pt) # defined in wls.jl

    # update
    mpvalues.xp = xp
    allactive = update_active_gridindices!(mpvalues, gridindices(F, grid, pt), spat)
    if allactive
        wᵢ, ∇wᵢ = values_gradients(F, grid, pt)
        mpvalues.N .= wᵢ
        mpvalues.∇N .= ∇wᵢ
    else
        M = zero(Mat{dim+1, dim+1, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w = value(F, grid, I, pt)
            P = [1; xi - xp]
            M += w * P ⊗ P
            mpvalues.N[i] = w
        end
        Minv = inv(M)
        C1 = Minv[1,1]
        C2 = @Tensor Minv[2:end,1]
        C3 = @Tensor Minv[2:end,2:end]
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w = mpvalues.N[i]
            mpvalues.N[i] = (C1 + C2 ⋅ (xi - xp)) * w
            mpvalues.∇N[i] = (C2 + C3 ⋅ (xi - xp)) * w
        end
    end

    mpvalues
end

@inline function Base.getindex(mpvalues::KernelCorrectionValues, i::Int)
    @_propagate_inbounds_meta
    KernelCorrectionValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.xp)
end
