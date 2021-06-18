struct BSplineValue{order, dim, T} <: ShapeValue{dim, T}
    F::BSpline{order, dim}
    N::Vector{T}
    dN::Vector{Vec{dim, T}}
end

function construct(::Type{T}, F::BSpline{order, dim}) where {order, dim, T}
    N = Vector{T}(undef, 0)
    dN = Vector{Vec{dim, T}}(undef, 0)
    BSplineValue(F, N, dN)
end

function reinit!(it::BSplineValue{<: Any, dim}, grid::AbstractGrid{dim}, indices::AbstractArray, x::Vec{dim}) where {dim}
    @boundscheck checkbounds(grid, indices)
    F = it.F
    resize!(it.N, length(indices))
    resize!(it.dN, length(indices))
    @inbounds for (j, I) in enumerate(view(CartesianIndices(grid), indices))
        xᵢ = grid[I]
        it.N[j], it.dN[j] = _value_gradient(F, x, xᵢ, gridsteps(grid), BSplinePosition(grid, I))
    end
    it
end

function _value(F::ShapeFunction, x::Vec{dim}, xᵢ::Vec{dim}, h::NTuple{dim}, pos) where {dim}
    ξ = (x - xᵢ) ./ h
    value(F, ξ, pos)
end

function _value_gradient(F::ShapeFunction, x::Vec, xᵢ::Vec, h::Tuple, pos)
    dv, v = gradient(x -> _value(F, x, xᵢ, h, pos), x, :all)
    v, dv
end
