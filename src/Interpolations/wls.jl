"""
    WLS(::Kernel)

WLS (weighted least squares) interpolation using the linear polynomial.

This interpolation function is used in the moving least squares material point method (MLS-MPM) [^MLSMPM],
but it is referred as the WLS in this library because the formulation is fundamentally WLS scheme.

[^MLSMPM]: [Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A., & Jiang, C. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. *ACM Transactions on Graphics (TOG)*, 37(4), 1-14.](https://doi.org/10.1145/3197517.3201293)
"""
struct WLS{K <: Kernel} <: Interpolation
    kernel::K
end

get_kernel(wls::WLS) = wls.kernel
gridspan(wls::WLS) = gridspan(get_kernel(wls))
@inline neighboringnodes(wls::WLS, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(wls), pt, mesh)

# general version
function update_property!(mp::MPValue{<: WLS}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)

    it = interpolation(mp)
    F = get_kernel(it)
    xₚ = getx(pt)

    M = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            xᵢ = mesh[i]
            w = mp.N[ip] = value(F, pt, mesh, i) * filter[i]
            p = [1; xᵢ-xₚ]
            w * p ⊗ p
        end
    end
    M⁻¹ = inv(M)

    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        wq = mp.N[ip] * (M⁻¹ ⋅ [1;xᵢ-xₚ])
        mp.N[ip] = wq[1]
        mp.∇N[ip] = @Tensor wq[2:end]
    end
end

# fast version for `LinearWLS(BSpline{order}())`
function update_property!(mp::MPValue{<: WLS{<: BSpline}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    if isnearbounds(mp)
        fast_update_property_nearbounds!(mp, pt, mesh, filter)
    else
        fast_update_property!(mp, pt, mesh)
    end
end

function fast_update_property!(mp::MPValue{<: WLS{<: BSpline}}, pt, mesh::CartesianMesh)
    indices = neighboringnodes(mp)
    F = get_kernel(interpolation(mp))
    xₚ = getx(pt)
    copyto!(mp.N, values(F, xₚ, mesh))

    # compute only diagonal entries
    D = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            w = mp.N[ip]
            xᵢ = mesh[i]
            mp.∇N[ip] = w * (xᵢ - xₚ)
            w * (xᵢ - xₚ) .* (xᵢ - xₚ)
        end
    end
    D⁻¹ = inv.(D)

    broadcast!(.*, mp.∇N, mp.∇N, D⁻¹)
end

function fast_update_property_nearbounds!(mp::MPValue{<: WLS{<: BSpline}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = neighboringnodes(mp)
    it = interpolation(mp)
    F = get_kernel(it)
    xₚ = getx(pt)

    M = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            xᵢ = mesh[i]
            w = mp.N[ip] = value(F, xₚ, mesh, i) * filter[i]
            p = [1; xᵢ-xₚ]
            w * p ⊗ p
        end
    end
    M⁻¹ = inv(M)

    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        wq = mp.N[ip] * (M⁻¹ ⋅ [1;xᵢ-xₚ])
        mp.N[ip] = wq[1]
        mp.∇N[ip] = @Tensor wq[2:end]
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", get_kernel(wls), ")")
