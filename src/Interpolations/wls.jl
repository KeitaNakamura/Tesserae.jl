struct WLS{K <: Kernel, P <: AbstractPolynomial} <: Interpolation
    kernel::K
    poly::P
end

WLS(k::Kernel) = WLS(k, LinearPolynomial())

get_kernel(wls::WLS) = wls.kernel
get_polynomial(wls::WLS) = wls.poly
gridspan(wls::WLS) = gridspan(get_kernel(wls))
@inline neighboringnodes(wls::WLS, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(wls), pt, mesh)

@inline function update_property!(mp::MPValue, it::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    update_property_general!(mp, it, pt, mesh, filter)
end

# a bit faster implementation for B-splines
@inline function update_property!(mp::MPValue, it::WLS{<: Union{BSpline{2}, BSpline{3}}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(mp)
    isnearbounds = size(mp.w) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_general!(mp, it, pt, mesh, filter)
    else
        kernel = get_kernel(it)
        @inbounds for ip in eachindex(indices)
            mp.w[ip] = value(kernel, pt, mesh, indices[ip])
        end
        update_property_after_moment_matrix!(mp, it, pt, mesh, moment_matrix_inv(kernel, mesh))
    end
end
@inline moment_matrix_inv(::BSpline{2}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 4/spacing(mesh)^2])
@inline moment_matrix_inv(::BSpline{3}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 3/spacing(mesh)^2])

@inline function update_property_general!(mp::MPValue, it::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = neighboringnodes(mp)
    kernel = get_kernel(it)
    poly = get_polynomial(it)
    xₚ = getx(pt)

    M = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            xᵢ = mesh[i]
            w = mp.w[ip] = value(kernel, pt, mesh, i) * filter[i]
            P = value(poly, xᵢ - xₚ)
            w * P ⊗ P
        end
    end

    update_property_after_moment_matrix!(mp, it, pt, mesh, inv(M))
end

@inline function update_property_after_moment_matrix!(mp::MPValue, it::WLS, pt, mesh::CartesianMesh, M⁻¹)
    indices = neighboringnodes(mp)
    poly = get_polynomial(it)
    xₚ = getx(pt)

    P₀, ∇P₀, ∇∇P₀, ∇∇∇P₀ = value(all, poly, zero(xₚ))
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = mp.w[ip]
        P = value(poly, xᵢ - xₚ)
        wq = w * (M⁻¹ ⋅ P)
        hasproperty(mp, :w)    && set_kernel_values!(mp, ip, (wq⋅P₀,))
        hasproperty(mp, :∇w)   && set_kernel_values!(mp, ip, (wq⋅P₀, wq⋅∇P₀))
        hasproperty(mp, :∇∇w)  && set_kernel_values!(mp, ip, (wq⋅P₀, wq⋅∇P₀, wq⋅∇∇P₀))
        hasproperty(mp, :∇∇∇w) && set_kernel_values!(mp, ip, (wq⋅P₀, wq⋅∇P₀, wq⋅∇∇P₀, wq⋅∇∇∇P₀))
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", get_kernel(wls), ", ", get_polynomial(wls), ")")
