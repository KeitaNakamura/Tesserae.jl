struct GIMP <: Kernel end

getsupportlength(::GIMP, l) = 1.0 .+ l # `l` must be normalized by `dx`

@pure nnodes(f::GIMP, ::Val{dim}) where {dim} = prod(nfill(3, Val(dim)))

@inline function value(::GIMP, ξ::Real, l::Real) # `l` is normalized radius
    ξ = abs(ξ)
    ξ < l   ? 1 - (ξ^2 + l^2) / 2l :
    ξ < 1-l ? 1 - ξ                :
    ξ < 1+l ? (1+l-ξ)^2 / 4l       : zero(ξ)
end

# `x` and `l` must be normalized by `dx`
@inline function Base.values(::GIMP, x::T, l::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    value.((GIMP(),), ξ, l)
end

# `x` and `l` must be normalized by `dx`
_gradient_GIMP(x, l) = gradient(x -> value(GIMP(), x, l), x, :all)
@generated function values_gradients(::GIMP, x::T, l::T) where {T <: Real}
    exps = [:(_gradient_GIMP(ξ[$i], l)) for i in 1:3]
    quote
        @_inline_meta
        V = Vec{3, T}
        x′ = fract(x - T(0.5))
        ξ = x′ .- V(-0.5, 0.5, 1.5)
        vals_grads = tuple($(exps...))
        Vec($([:(vals_grads[$i][2]) for i in 1:3]...)), Vec($([:(vals_grads[$i][1]) for i in 1:3]...))
    end
end

@generated function values_gradients(::GIMP, x::Vec{dim}, l::Vec{dim}) where {dim}
    exps = [:(values_gradients(GIMP(), x[$i], l[$i])) for i in 1:dim]
    derivs = map(1:dim) do i
        x = [d == i ? :(grads[$d]) : :(vals[$d]) for d in 1:dim]
        :(Tuple(otimes($(x...))))
    end
    quote
        @_inline_meta
        vals_grads = tuple($(exps...))
        vals = getindex.(vals_grads, 1)
        grads = getindex.(vals_grads, 2)
        Tuple(otimes(vals...)), Vec{dim}.($(derivs...))
    end
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
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    mpvalues.x = x
    dx⁻¹ = gridsteps_inv(grid)
    update_gridindices!(mpvalues, neighboring_nodes(grid, x, getsupportlength(F, r .* dx⁻¹)), spat)
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
