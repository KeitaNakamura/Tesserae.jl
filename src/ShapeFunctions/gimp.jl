struct GIMP <: ShapeFunction end

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


struct GIMPValues{dim, T, L} <: ShapeValues{dim, T}
    F::GIMP
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    inds::MVector{L, Index{dim}}
    len::Base.RefValue{Int}
end

function GIMPValues{dim, T, L}() where {dim, T, L}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    inds = MVector{L, Index{dim}}(undef)
    GIMPValues(GIMP(), N, ∇N, inds, Ref(0))
end

function ShapeValues{dim, T}(F::GIMP) where {dim, T}
    L = nnodes(F, Val(dim))
    GIMPValues{dim, T, L}()
end

function update!(it::GIMPValues{dim}, grid::Grid{dim}, x::Vec{dim}, r::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    it.N .= zero(it.N)
    it.∇N .= zero(it.∇N)
    F = it.F
    update_gridindices!(it, grid, neighboring_nodes(grid, x, support_length(F)), spat)
    @inbounds @simd for i in 1:length(it)
        I = it.inds[i]
        xᵢ = grid[I]
        it.∇N[i], it.N[i] = gradient(x, :all) do x
            @_inline_meta
            ξ = (x - xᵢ) ./ gridsteps(grid)
            value(F, ξ, r ./ gridsteps(grid))
        end
    end
    it
end

struct GIMPValue{dim, T}
    N::T
    ∇N::Vec{dim, T}
    index::Index{dim}
end

@inline function Base.getindex(it::GIMPValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(it.N[i], it.∇N[i], it.inds[i])
end
