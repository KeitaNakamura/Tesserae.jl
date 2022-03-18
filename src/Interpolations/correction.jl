struct KernelCorrection{K <: Kernel} <: Interpolation
end
@pure KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

@pure getkernelfunction(::KernelCorrection{K}) where {K} = K()
getsupportlength(c::KernelCorrection, args...) = getsupportlength(getkernelfunction(c), args...)


struct KernelCorrectionValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    I::Index{dim}
    x::Vec{dim, T}
end

mutable struct KernelCorrectionValues{K, dim, T, nnodes} <: MPValues{dim, T, KernelCorrectionValue{dim, T}}
    F::KernelCorrection{K}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    gridindices::MVector{nnodes, Index{dim}}
    x::Vec{dim, T}
    len::Int
end

function KernelCorrectionValues{K, dim, T, nnodes}() where {K, dim, T, nnodes}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    gridindices = MVector{nnodes, Index{dim}}(undef)
    x = zero(Vec{dim, T})
    KernelCorrectionValues(KernelCorrection(K()), N, ∇N, gridindices, x, 0)
end

function MPValues{dim, T}(c::KernelCorrection{K}) where {dim, T, K}
    L = nnodes(K(), Val(dim))
    KernelCorrectionValues{K, dim, T, L}()
end

getkernelfunction(c::KernelCorrectionValues) = getkernelfunction(c.F)

function _update!(mpvalues::KernelCorrectionValues{<: Any, dim, T}, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}, inds, args...) where {dim, T}
    F = getkernelfunction(mpvalues)
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    mpvalues.x = x

    dx⁻¹ = gridsteps_inv(grid)
    iscompleted = update_gridindices!(mpvalues, inds, spat)
    if iscompleted
        wᵢ, ∇wᵢ = values_gradients(F, x .* dx⁻¹, args...)
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            mpvalues.N[i] = wᵢ[i]
            mpvalues.∇N[i] = ∇wᵢ[i] .* dx⁻¹
        end
    else
        A = zero(Mat{dim, dim, T})
        β = zero(Vec{dim, T})
        A′ = zero(Mat{dim, dim, T})
        β′ = zero(Vec{dim, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            ξ = (x - xᵢ) .* dx⁻¹
            ∇w, w = gradient(ξ -> value(F, ξ, args...), ξ, :all)
            ∇w = ∇w .* dx⁻¹
            A += w * (xᵢ - x) ⊗ (xᵢ - x)
            β += w * (xᵢ - x)
            A′ += ∇w ⊗ (xᵢ - x)
            β′ += ∇w
            mpvalues.N[i] = w
            mpvalues.∇N[i] = ∇w
        end
        A⁻¹ = inv(A)
        β = -(A⁻¹ ⋅ β)
        A′⁻¹ = inv(A′)
        β′ = -(A′⁻¹ ⋅ β′)
        α = zero(T)
        α′ = zero(Mat{dim, dim, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            w = mpvalues.N[i]
            ∇w = mpvalues.∇N[i]
            α += w * (1 + β ⋅ (xᵢ - x))
            α′ += (xᵢ ⊗ ∇w) * (1 + β′ ⋅ (xᵢ - x))
        end
        α = inv(α)
        α′ = inv(α′)
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            w = mpvalues.N[i]
            ∇w = mpvalues.∇N[i]
            mpvalues.N[i] = w * α * (1 + β ⋅ (xᵢ - x))
            mpvalues.∇N[i] = ∇w ⋅ α′ * (1 + β′ ⋅ (xᵢ - x))
        end
    end

    mpvalues
end

function update!(mpvalues::KernelCorrectionValues, grid::Grid, x::Vec, spat::AbstractArray{Bool})
    F = getkernelfunction(mpvalues)
    dx⁻¹ = gridsteps_inv(grid)
    _update!(mpvalues, grid, x, spat, neighboring_nodes(grid, x, getsupportlength(F)))
end

function update!(mpvalues::KernelCorrectionValues{GIMP}, grid::Grid, x::Vec, r::Vec, spat::AbstractArray{Bool})
    F = getkernelfunction(mpvalues)
    dx⁻¹ = gridsteps_inv(grid)
    rdx⁻¹ = r.*dx⁻¹
    _update!(mpvalues, grid, x, spat, neighboring_nodes(grid, x, getsupportlength(F, rdx⁻¹)), rdx⁻¹)
end

@inline function Base.getindex(mpvalues::KernelCorrectionValues, i::Int)
    @_propagate_inbounds_meta
    KernelCorrectionValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.gridindices[i], mpvalues.x)
end
