"""
    WLS(kernel)

`WLS` performs a local weighted least squares fit for the `kernel`.
This results in the same kernel used in moving least squares MPM[^MLSMPM].
`kernel` is one of [`BSpline`](@ref) and [`uGIMP`](@ref).

[^MLSMPM]: [Hu, Y., Fang, Y., Ge, Z., Qu, Z., Zhu, Y., Pradhana, A. and Jiang, C., 2018. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Transactions on Graphics (TOG), 37(4), pp.1-14.](https://doi.org/10.1145/3197517.3201293)
"""
struct WLS{K <: Kernel, P <: Polynomial} <: Interpolation
    kernel::K
    poly::P
end

WLS(k::Kernel) = WLS(k, Polynomial(Linear()))

get_kernel(wls::WLS) = wls.kernel
get_polynomial(wls::WLS) = wls.poly
gridspan(wls::WLS) = gridspan(get_kernel(wls))
@inline neighboringnodes(wls::WLS, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(wls), pt, mesh)

@inline function update_property!(mp::MPValue, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    update_property_general!(mp, wls, pt, mesh, filter)
end

# a bit faster implementation for B-splines
@inline function update_property!(mp::MPValue, wls::WLS{<: Union{BSpline{Quadratic}, BSpline{Cubic}}, <: Polynomial{Linear}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    isnearbounds = size(values(mp,1)) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_general!(mp, wls, pt, mesh, filter)
    else
        kernel = get_kernel(wls)
        @inbounds for ip in eachindex(indices)
            values(mp,1)[ip] = value(kernel, pt, mesh, indices[ip])
        end
        update_property_after_moment_matrix!(mp, wls, pt, mesh, moment_matrix_inv(kernel, mesh))
    end
end
@inline moment_matrix_inv(::BSpline{Quadratic}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 4/spacing(mesh)^2])
@inline moment_matrix_inv(::BSpline{Cubic}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 3/spacing(mesh)^2])

@inline function update_property_general!(mp::MPValue, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = neighboringnodes(mp)
    kernel = get_kernel(wls)
    poly = get_polynomial(wls)
    xₚ = getx(pt)

    M = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            xᵢ = mesh[i]
            w = values(mp,1)[ip] = value(kernel, pt, mesh, i) * filter[i]
            P = value(poly, xᵢ - xₚ)
            w * P ⊗ P
        end
    end

    update_property_after_moment_matrix!(mp, wls, pt, mesh, inv(M))
end

@inline function update_property_after_moment_matrix!(mp::MPValue, wls::WLS, pt, mesh::CartesianMesh, M⁻¹)
    indices = neighboringnodes(mp)
    poly = get_polynomial(wls)
    xₚ = getx(pt)

    P₀__ = values(derivative_order(mp), poly, zero(xₚ))
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = values(mp,1)[ip]
        P = value(poly, xᵢ - xₚ)
        wq = w * (M⁻¹ * P)
        set_values!(mp, ip, map(P₀->wq⊡P₀, P₀__))
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", get_kernel(wls), ", ", get_polynomial(wls), ")")
