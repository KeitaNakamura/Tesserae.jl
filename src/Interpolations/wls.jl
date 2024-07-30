struct WLS{K <: Kernel, P <: AbstractPolynomial} <: Interpolation
    kernel::K
    poly::P
end

WLS(k::Kernel) = WLS(k, LinearPolynomial())

get_kernel(wls::WLS) = wls.kernel
get_polynomial(wls::WLS) = wls.poly
gridspan(wls::WLS) = gridspan(get_kernel(wls))
@inline neighboringnodes(wls::WLS, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(wls), pt, mesh)

# implementation is not fast
@inline function update_property!(mp::MPValue, it::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
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
    M⁻¹ = inv(M)

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
