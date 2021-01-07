abstract type ShapeValue{dim, T} <: AbstractCollection{1, ScalVec{dim, T}} end

"""
    construct(::ShapeFunction)
    construct(::Type{T}, ::ShapeFunction)

Construct object storing value of `ShapeFunction`.

# Examples
```jldoctest
julia> N = construct(QuadraticBSpline(dim = 2));

julia> reinit!(N, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(N)
1.0

julia> sum(∇, N)
2-element Tensor{1,2,Float64,2}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
construct(F::ShapeFunction{dim}) where {dim} = construct(Float64, F)

"""
    reinit!(::ShapeValue, grid::AbstractGrid, x::Vec)
    reinit!(::ShapeValue, grid::AbstractGrid, indices::AbstractArray, x::Vec)

Reinitialize value of shape function at `x` with each `grid` node.

# Examples
```jldoctest
julia> N = construct(QuadraticBSpline(dim = 2));

julia> reinit!(N, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(N)
1.0

julia> reinit!(N, Grid(0:3, 0:3), CartesianIndices((1:2, 1:2)), Vec(1, 1));

julia> sum(N)
0.765625
```
"""
reinit!

function reinit!(it::ShapeValue, grid::AbstractGrid, x::Vec)
    reinit!(it, grid, eachindex(grid), x)
end

Base.length(it::ShapeValue) = length(it.N)

@inline function Base.getindex(it::ShapeValue, i::Int)
    @_propagate_inbounds_meta
    ScalVec(it.N[i], it.dN[i])
end


struct VectorValue{dim, T, IT <: ShapeValue{dim, T}, M} <: AbstractCollection{1, VecTensor{dim, T, M}}
    Ni::IT
end

function Base.vec(Ni::ShapeValue{dim, T}) where {dim, T}
    VectorValue{dim, T, typeof(Ni), dim^2}(Ni)
end

Base.length(it::VectorValue{dim}) where {dim} = dim * length(it.Ni)

@inline function Base.getindex(it::VectorValue{dim}, j::Int) where {dim}
    @boundscheck checkbounds(it, j)
    i, d = divrem(j - 1, dim) .+ 1
    @inbounds begin
        ei = eᵢ(Vec{dim, Int}, d)
        N = it.Ni[i]
    end
    VecTensor(ei * N, ei ⊗ ∇(N))
end
