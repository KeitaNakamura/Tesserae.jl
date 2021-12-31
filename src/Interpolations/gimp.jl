struct GIMP <: Kernel end

support_length(::GIMP) = 2.0
active_length(::GIMP, l) = 1.0 .+ l # `l` must be normalized by `dx`

@pure nnodes(f::GIMP, ::Val{dim}) where {dim} = prod(nfill(Int(2*support_length(f)), Val(dim)))

@inline function value(::GIMP, ξ::Real, l::Real) # `l` is normalized radius
    ξ = abs(ξ)
    ξ < l   ? 1 - (ξ^2 + l^2) / 2l :
    ξ < 1-l ? 1 - ξ                :
    ξ < 1+l ? (1+l-ξ)^2 / 4l       : zero(ξ)
end

@generated function value(f::GIMP, ξ::Vec{dim, T}, l::Vec{dim}) where {dim, T}
    exps = [:(value(f, ξ[$i], l[$i])) for i in 1:dim]
    quote
        @_inline_meta
        *($(exps...))
    end
end


mutable struct GIMPValues{dim, T, L} <: MPValues{dim, T}
    F::GIMP
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    gridindices::MVector{L, Index{dim}}
    x::Vec{dim, T}
    len::Int
end

function GIMPValues{dim, T, L}() where {dim, T, L}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    gridindices = MVector{L, Index{dim}}(undef)
    x = zero(Vec{dim, T})
    GIMPValues(GIMP(), N, ∇N, gridindices, x, 0)
end

function MPValues{dim, T}(F::GIMP) where {dim, T}
    L = nnodes(F, Val(dim))
    GIMPValues{dim, T, L}()
end

function update!(mpvalues::GIMPValues{dim}, grid::Grid{dim}, x::Vec{dim}, r::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    F = mpvalues.F
    mpvalues.N .= zero(mpvalues.N)
    mpvalues.∇N .= zero(mpvalues.∇N)
    mpvalues.x = x
    update_gridindices!(mpvalues, grid, x, spat)
    dx⁻¹ = gridsteps_inv(grid)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        mpvalues.∇N[i], mpvalues.N[i] = gradient(x, :all) do x
            @_inline_meta
            ξ = (x - xᵢ) .* dx⁻¹
            value(F, ξ, r .* dx⁻¹)
        end
    end
    mpvalues
end

struct GIMPValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    I::Index{dim}
    x::Vec{dim, T}
end

@inline function Base.getindex(mpvalues::GIMPValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.gridindices[i], mpvalues.x)
end
