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
kernel_support(wls::WLS) = kernel_support(get_kernel(wls))
@inline neighboringnodes(wls::WLS, pt, mesh::CartesianMesh) = neighboringnodes(get_kernel(wls), pt, mesh)

@inline function update_property!(iw::InterpolationWeight, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    update_property_general!(iw, wls, pt, mesh, filter)
end

# a bit faster implementation for B-splines
@inline function update_property!(iw::InterpolationWeight, wls::WLS{<: Union{BSpline{Quadratic}, BSpline{Cubic}}, <: Polynomial{Linear}}, pt, mesh::CartesianMesh, filter::AbstractArray{Bool} = Trues(size(mesh)))
    indices = neighboringnodes(iw)
    isnearbounds = size(values(iw,1)) != size(indices) || !alltrue(filter, indices)
    if isnearbounds
        update_property_general!(iw, wls, pt, mesh, filter)
    else
        kernel = get_kernel(wls)
        @inbounds for ip in eachindex(indices)
            values(iw,1)[ip] = value(kernel, pt, mesh, indices[ip])
        end
        update_property_after_moment_matrix!(iw, wls, pt, mesh, moment_matrix_inv(kernel, mesh))
    end
end
@inline moment_matrix_inv(::BSpline{Quadratic}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 4/spacing(mesh)^2])
@inline moment_matrix_inv(::BSpline{Cubic}, mesh::CartesianMesh{dim}) where {dim} = diagm([1; ones(Vec{dim,Int}) * 3/spacing(mesh)^2])

@inline function update_property_general!(iw::InterpolationWeight, wls::WLS, pt, mesh::CartesianMesh, filter::AbstractArray{Bool})
    indices = neighboringnodes(iw)
    kernel = get_kernel(wls)
    poly = get_polynomial(wls)
    xₚ = getx(pt)

    M = fastsum(eachindex(indices)) do ip
        @inbounds begin
            i = indices[ip]
            xᵢ = mesh[i]
            w = values(iw,1)[ip] = value(kernel, pt, mesh, i) * filter[i]
            P = value(poly, xᵢ - xₚ)
            w * P ⊗ P
        end
    end

    update_property_after_moment_matrix!(iw, wls, pt, mesh, inv(M))
end

@inline function update_property_after_moment_matrix!(iw::InterpolationWeight, wls::WLS, pt, mesh::CartesianMesh, M⁻¹)
    indices = neighboringnodes(iw)
    poly = get_polynomial(wls)
    xₚ = getx(pt)

    P₀__ = values(derivative_order(iw), poly, zero(xₚ))
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        xᵢ = mesh[i]
        w = values(iw,1)[ip]
        P = value(poly, xᵢ - xₚ)
        wq = w * (M⁻¹ * P)
        set_values!(iw, ip, map(P₀->wq⊡P₀, P₀__))
    end
end

@inline function update_property!(iw::InterpolationWeight, wls::WLS{<: Union{BSpline{Quadratic}, BSpline{Cubic}, BSpline{Quartic}, BSpline{Quintic}}, Polynomial{MultiLinear}}, pt, mesh::CartesianMesh{dim}, filter::AbstractArray{Bool} = Trues(size(mesh))) where {dim}
    if filter isa Trues
        # For MultiLinear, we can decompose into axis-wise Linear interpolations.
        # If the problem is 1D, MultiLinear == Linear, so use the direct fast path.
        wls_1d = WLS(get_kernel(wls), Polynomial(Linear()))
        if dim == 1
            update_property!(iw, wls_1d, pt, mesh, filter)
        else
            # For dim > 1: decompose into 1D Linear along each axis,
            # compute axis-wise contribution, then combine by tensor product.
            order = derivative_order(iw)
            vals′ = ntuple(Val(dim)) do d
                T = eltype(values(iw, 1))
                mesh′ = axismesh(mesh, d)
                prop′ = create_property(MArray, Vec{1,T}, wls_1d; derivative=order, name=Val(propertynames(iw)[1]))
                indices′ = CartesianIndices((neighboringnodes(iw).indices[d],))
                iw′ = InterpolationWeight(wls_1d, prop′, Scalar(indices′))
                update_property!(iw′, wls_1d, Vec(getx(pt)[d]), mesh′)
                # Get scalar value from Vec{1} for each property.
                _extract_scalar_values(order, iw′)
            end
            # Combine axis-wise results into MultiLinear tensor product.
            set_values!(iw, _prod_each_dimension(order, vals′))
        end
    else
        # Fallback for masked cases: use general method.
        update_property_general!(iw, wls, pt, mesh, filter)
    end
end
@generated function _extract_scalar_values(::Order{k}, iw) where {k}
    quote
        @_inline_meta
        @ntuple $(k+1) a -> map(only, Tuple(values(iw, a)))
    end
end
@generated function _prod_each_dimension(::Order{k}, vals) where {k}
    quote
        @_inline_meta
        @ntuple $(k+1) a -> prod_each_dimension(Order(a-1), vals...)
    end
end

Base.show(io::IO, wls::WLS) = print(io, WLS, "(", get_kernel(wls), ", ", get_polynomial(wls), ")")
