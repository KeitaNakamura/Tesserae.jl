struct KernelCorrection{K <: Kernel} <: Interpolation
end
@pure KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

@pure getkernelfunction(::KernelCorrection{K}) where {K} = K()

@inline function neighbornodes(x::KernelCorrection, grid::Grid, pt)
    neighbornodes(getkernelfunction(x), grid, pt)
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
    L = getnnodes(K(), Val(dim))
    KernelCorrectionValues{K, dim, T, L}()
end

getkernelfunction(c::KernelCorrectionValues) = getkernelfunction(c.F)

function update!(mpvalues::KernelCorrectionValues{<: Any, dim, T}, grid::Grid{<: Any, dim}, pt, spat::AbstractArray{Bool, dim}) where {dim, T}
    # reset
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)

    F = getkernelfunction(mpvalues)
    xp = getx(pt) # defined in wls.jl

    # update
    mpvalues.xp = xp
    allactive = update_active_gridindices!(mpvalues, neighbornodes(F, grid, pt), spat)
    if allactive
        wᵢ, ∇wᵢ = values_gradients(F, grid, pt)
        mpvalues.N .= wᵢ
        mpvalues.∇N .= ∇wᵢ
    else
        A = zero(Mat{dim, dim, T})
        β = zero(Vec{dim, T})
        A′ = zero(Mat{dim, dim, T})
        β′ = zero(Vec{dim, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w, ∇w = value_gradient(F, grid, I, pt)
            A += w * (xi - xp) ⊗ (xi - xp)
            β += w * (xi - xp)
            A′ += ∇w ⊗ (xi - xp)
            β′ += ∇w
            mpvalues.N[i] = w
            mpvalues.∇N[i] = ∇w
        end
        β = inv(A) ⋅ β
        β′ = inv(A′) ⋅ β′
        α = zero(T)
        α′ = zero(Mat{dim, dim, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            mpvalues.N[i] *= 1 + β ⋅ (xp - xi)
            mpvalues.∇N[i] *= 1 + β′ ⋅ (xp - xi)
            α += mpvalues.N[i]
            α′ += xi ⊗ mpvalues.∇N[i]
        end
        @. mpvalues.N = mpvalues.N * $inv(α)
        @. mpvalues.∇N = mpvalues.∇N ⋅ $inv(α′)
    end

    mpvalues
end

@inline function Base.getindex(mpvalues::KernelCorrectionValues, i::Int)
    @_propagate_inbounds_meta
    KernelCorrectionValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.xp)
end
