struct KernelCorrection{K <: Kernel} <: Interpolation
end
KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

get_kernel(::KernelCorrection{K}) where {K} = K()
@inline neighbornodes(kc::KernelCorrection, lattice::Lattice, pt) = neighbornodes(get_kernel(kc), lattice, pt)

function MPValuesInfo{dim, T}(itp::KernelCorrection) where {dim, T}
    dims = nfill(gridsize(get_kernel(itp)), Val(dim))
    values = (; N=zero(T), ∇N=zero(Vec{dim, T}))
    sizes = (dims, dims)
    MPValuesInfo{dim, T}(values, sizes)
end

# general version
@inline function update_mpvalues!(mp::SubMPValues, itp::KernelCorrection, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, isfullyinside = neighbornodes(itp, lattice, pt)

    if isfullyinside && @inbounds alltrue(sppat, indices)
        @inbounds for (j, i) in pairs(IndexCartesian(), indices)
            mp.N[j], mp.∇N[j] = value_gradient(get_kernel(itp), lattice, i, pt)
        end
    else
        update_mpvalue_nearbounds!(mp, itp, lattice, sppat, indices, pt)
    end

    indices
end

# fast version for B-spline kernels
@inline function update_mpvalues!(mp::SubMPValues, itp::KernelCorrection{<: BSpline}, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, isfullyinside = neighbornodes(itp, lattice, pt)

    if isfullyinside && @inbounds alltrue(sppat, indices)
        values_gradients!(mp.N, mp.∇N, get_kernel(itp), lattice, pt)
    else
        update_mpvalue_nearbounds!(mp, itp, lattice, sppat, indices, pt)
    end

    indices
end

@inline function update_mpvalue_nearbounds!(mp::SubMPValues{dim, T}, itp::KernelCorrection, lattice::Lattice, sppat::AbstractArray{Bool}, indices, pt) where {dim, T}
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
