struct KernelCorrection{K <: Kernel} <: Interpolation
end
KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

get_kernel(::KernelCorrection{K}) where {K} = K()
@inline neighbornodes(kc::KernelCorrection, lattice::Lattice, pt) = neighbornodes(get_kernel(kc), lattice, pt)

struct KernelCorrectionValue{dim, T, K} <: MPValue{dim, T}
    itp::KernelCorrection{K}
    N::Vector{T}
    ∇N::Vector{Vec{dim, T}}
end

function MPValue{dim, T}(itp::KernelCorrection{K}) where {dim, T, K}
    N = Vector{T}(undef, 0)
    ∇N = Vector{Vec{dim, T}}(undef, 0)
    KernelCorrectionValue(itp, N, ∇N)
end

num_nodes(mp::KernelCorrectionValue) = length(mp.N)
@inline shape_value(mp::KernelCorrectionValue, j::Int) = (@_propagate_inbounds_meta; mp.N[j])
@inline shape_gradient(mp::KernelCorrectionValue, j::Int) = (@_propagate_inbounds_meta; mp.∇N[j])

# general version
@inline function update_mpvalue!(mp::KernelCorrectionValue, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, isfullyinside = neighbornodes(mp.itp, lattice, pt)

    n = length(indices)
    resize!(mp.N, n)
    resize!(mp.∇N, n)

    if isfullyinside && @inbounds alltrue(sppat, indices)
        @inbounds for (j, i) in pairs(IndexLinear(), indices)
            mp.N[j], mp.∇N[j] = value_gradient(get_kernel(mp.itp), lattice, i, pt)
        end
    else
        update_mpvalue_nearbounds!(mp, lattice, sppat, indices, pt)
    end

    indices
end

# fast version for B-spline kernels
@inline function update_mpvalue!(mp::KernelCorrectionValue{<: Any, <: Any, <: BSpline}, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    indices, isfullyinside = neighbornodes(mp.itp, lattice, pt)

    n = length(indices)
    resize!(mp.N, n)
    resize!(mp.∇N, n)

    if isfullyinside && @inbounds alltrue(sppat, indices)
        fast_update_mpvalue!(mp, lattice, sppat, indices, pt)
    else
        update_mpvalue_nearbounds!(mp, lattice, sppat, indices, pt)
    end

    indices
end

@inline function fast_update_mpvalue!(mp::KernelCorrectionValue{<: Any, T}, lattice::Lattice, sppat::AbstractArray{Bool}, indices, pt) where {T}
    N = mp.N
    ∇N = reinterpret(reshape, T, mp.∇N)
    values_gradients!(N, ∇N, get_kernel(mp.itp), lattice, pt)
end

@inline function update_mpvalue_nearbounds!(mp::KernelCorrectionValue{dim, T}, lattice::Lattice, sppat::AbstractArray{Bool}, indices, pt) where {dim, T}
    F = get_kernel(mp.itp)
    xp = getx(pt)
    M = zero(Mat{dim+1, dim+1, T})
    @inbounds for (j, i) in pairs(IndexLinear(), indices)
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
    @inbounds for (j, i) in pairs(IndexLinear(), indices)
        xi = lattice[i]
        w = mp.N[j]
        mp.N[j] = (C1 + C2 ⋅ (xi - xp)) * w
        mp.∇N[j] = (C2 + C3 ⋅ (xi - xp)) * w
    end
end
