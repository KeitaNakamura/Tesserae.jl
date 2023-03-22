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
gridsize(kc::KernelCorrection) = gridsize(get_kernel(kc))
@inline neighbornodes(kc::KernelCorrection, lattice::Lattice, pt) = neighbornodes(get_kernel(kc), lattice, pt)

function MPValuesInfo{dim, T}(itp::KernelCorrection) where {dim, T}
    dims = nfill(gridsize(itp), Val(dim))
    values = (; N=zero(T), ∇N=zero(Vec{dim, T}))
    sizes = (dims, dims)
    MPValuesInfo{dim, T}(values, sizes)
end

# general version
@inline function update!(mp::SubMPValues, itp::KernelCorrection, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, isfullyinside = neighbornodes(itp, lattice, pt)

    if isfullyinside && @inbounds alltrue(sppat, indices)
        @inbounds for (j, i) in pairs(IndexCartesian(), indices)
            mp.N[j], mp.∇N[j] = value_gradient(get_kernel(itp), lattice, i, pt)
        end
    else
        update_mpvalues_nearbounds!(mp, itp, lattice, sppat, indices, pt)
    end

    set_neighbornodes!(mp, indices)
end

# fast version for B-spline kernels
@inline function update!(mp::SubMPValues, itp::KernelCorrection{<: BSpline}, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, isfullyinside = neighbornodes(itp, lattice, pt)

    if isfullyinside && @inbounds alltrue(sppat, indices)
        values_gradients!(mp.N, mp.∇N, get_kernel(itp), lattice, pt)
    else
        update_mpvalues_nearbounds!(mp, itp, lattice, sppat, indices, pt)
    end

    set_neighbornodes!(mp, indices)
end

@inline function update_mpvalues_nearbounds!(mp::SubMPValues{dim, T}, itp::KernelCorrection, lattice::Lattice, sppat::AbstractArray{Bool}, indices, pt) where {dim, T}
    F = get_kernel(itp)
    xp = getx(pt)
    M = zero(Mat{dim+1, dim+1, T})
    @inbounds for (j, i) in pairs(IndexCartesian(), indices)
        xi = lattice[i]
        w = value(F, lattice, i, pt) * sppat[i]
        P = [1; xi - xp]
        M += w * P ⊗ P
        mp.N[j] = w
    end
    Minv = inv(M)
    C1 = Minv[1,1]
    C2 = @Tensor Minv[2:end,1]
    C3 = @Tensor Minv[2:end,2:end]
    @inbounds for (j, i) in pairs(IndexCartesian(), indices)
        xi = lattice[i]
        w = mp.N[j]
        mp.N[j] = (C1 + C2 ⋅ (xi - xp)) * w
        mp.∇N[j] = (C2 + C3 ⋅ (xi - xp)) * w
    end
end
