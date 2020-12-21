abstract type Interpolation{dim} end

"""
    construct(::ShapeFunction)
    construct(::Type{T}, ::ShapeFunction)

Construct interpolation object for `ShapeFunction`.

# Examples
```jldoctest
julia> it = construct(QuadraticBSpline(dim = 2));

julia> reinit!(it, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(i -> shape_value(it, i), 1:nvalues(it))
1.0

julia> sum(i -> shape_gradient(it, i), 1:nvalues(it))
2-element Tensor{1,2,Float64,2}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
construct(F::ShapeFunction{dim}) where {dim} = construct(Float64, F)::Interpolation{dim}

"""
    reinit!(::Interpolation, grid::AbstractGrid, x::Vec)
    reinit!(::Interpolation, grid::AbstractGrid, indices::AbstractArray, x::Vec)

Reinitialize interpolation object at `x` with each `grid` node.

# Examples
```jldoctest
julia> it = construct(QuadraticBSpline(dim = 2));

julia> reinit!(it, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(i -> shape_value(it, i), 1:nvalues(it))
1.0

julia> reinit!(it, Grid(0:3, 0:3), CartesianIndices((1:2, 1:2)), Vec(1, 1));

julia> sum(i -> shape_value(it, i), 1:nvalues(it))
0.765625
```
"""
reinit!

function reinit!(it::Interpolation, grid::AbstractGrid, x::Vec)
    reinit!(it, grid, eachindex(grid), x)
end

getshapefunction(it::Interpolation{dim}) where {dim} = it.F::ShapeFunction{dim}

nvalues(it::Interpolation) = length(it.N)
nvalues(::Type{Vec}, it::Interpolation{dim}) where {dim} = nvalues(it) * dim

shape_value(it::Interpolation, i::Int) = (@_propagate_inbounds_meta; it.N[i])

function shape_value(::Type{Vec}, it::Interpolation{dim}, j::Int) where {dim}
    @boundscheck 0 < j ≤ nvalues(Vec, it) || throw(ArgumentError("index $j is out of range, nvalues(Vec, ::Interpolation) = $(nvalues(Vec, it))"))
    i, d = divrem(j - 1, dim) .+ 1
    @inbounds eᵢ(Vec{dim, Int}, d) * shape_value(it, i)
end

shape_gradient(it::Interpolation, i::Int) = (@_propagate_inbounds_meta; it.dN[i])

function shape_gradient(::Type{Vec}, it::Interpolation{dim}, j::Int) where {dim}
    @boundscheck 0 < j ≤ nvalues(Vec, it) || throw(ArgumentError("index $j is out of range, nvalues(Vec, ::Interpolation) = $(nvalues(Vec, it))"))
    i, d = divrem(j - 1, dim) .+ 1
    @inbounds eᵢ(Vec{dim, Int}, d) ⊗ shape_gradient(it, i)
end

function shape_symmetricgradient(::Type{Vec}, it::Interpolation, j::Int)
    @_propagate_inbounds_meta
    symmetric(shape_gradient(Vec, it, j))
end
