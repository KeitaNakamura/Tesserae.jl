"""
    KernelCorrection(::Kernel)

Kernel correction method [^KC].

This modifies kernels to achive stable simulations near boundaries.
Using kernel correction is preferred for [`QuadraticBSpline`](@ref) and [`CubicBSpline`](@ref),
and almost necessary for [`AffineTransfer`](@ref) and [`TaylorTransfer`](@ref).

[^KC]: [Nakamura, K., Matsumura, S., & Mizutani, T. (2023). Taylor particle-in-cell transfer and kernel correction for material point method. *Computer Methods in Applied Mechanics and Engineering*, 403, 115720.](https://doi.org/10.1016/j.cma.2022.115720)
"""
struct KernelCorrection{K <: Kernel} <: Interpolation
end
KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

get_kernel(::KernelCorrection{K}) where {K} = K()
gridsize(kc::KernelCorrection, ::Val{dim}) where {dim} = gridsize(get_kernel(kc), Val(dim))
@inline neighbornodes(kc::KernelCorrection, lattice::Lattice, pt) = neighbornodes(get_kernel(kc), lattice, pt)

function MPValuesInfo{dim, T}(itp::KernelCorrection) where {dim, T}
    dims = gridsize(itp, Val(dim))
    values = (; N=zero(T), ∇N=zero(Vec{dim, T}))
    sizes = (dims, dims)
    MPValuesInfo{dim, T}(values, sizes)
end

# general version
@inline function update_mpvalues!(mp::SubMPValues, itp::KernelCorrection, lattice::Lattice, spy::AbstractArray{Bool}, pt)
    if isnearbounds(mp)
        update_mpvalues_nearbounds!(mp, itp, lattice, spy, pt)
    else
        indices = neighbornodes(mp)
        @inbounds @simd for j in CartesianIndices(indices)
            i = indices[j]
            mp.N[j], mp.∇N[j] = value_gradient(get_kernel(itp), lattice, i, pt)
        end
    end
end

# fast version for B-spline kernels
@inline function update_mpvalues!(mp::SubMPValues, itp::KernelCorrection{<: BSpline}, lattice::Lattice, spy::AbstractArray{Bool}, pt)
    if isnearbounds(mp)
        update_mpvalues_nearbounds!(mp, itp, lattice, spy, pt)
    else
        values_gradients!(mp.N, mp.∇N, get_kernel(itp), lattice, pt)
    end
end

@inline function update_mpvalues_nearbounds!(mp::SubMPValues{dim, T}, itp::KernelCorrection, lattice::Lattice, spy::AbstractArray{Bool}, pt) where {dim, T}
    indices = neighbornodes(mp)
    F = get_kernel(itp)
    xₚ = getx(pt)
    M = zero(Mat{dim+1, dim+1, T})
    @inbounds for j in CartesianIndices(indices)
        i = indices[j]
        xᵢ = lattice[i]
        w = value(F, lattice, i, pt) * spy[i]
        P = [1; xᵢ - xₚ]
        M += w * P ⊗ P
        mp.N[j] = w
    end
    M⁻¹ = inv(M)
    C₁ = M⁻¹[1,1]
    C₂ = @Tensor M⁻¹[2:end,1]
    C₃ = @Tensor M⁻¹[2:end,2:end]
    @inbounds for j in CartesianIndices(indices)
        i = indices[j]
        xᵢ = lattice[i]
        w = mp.N[j]
        mp.N[j] = (C₁ + C₂ ⋅ (xᵢ - xₚ)) * w
        mp.∇N[j] = (C₂ + C₃ ⋅ (xᵢ - xₚ)) * w
    end
end

Base.show(io::IO, kc::KernelCorrection) = print(io, KernelCorrection, "(", get_kernel(kc), ")")
