abstract type ShapeValues{dim, T} <: AbstractVector{T} end

Base.size(x::ShapeValues) = (x.len[],)

"""
    Poingr.ShapeValues{dim}(::ShapeFunction)
    Poingr.ShapeValues{dim, T}(::ShapeFunction)

Construct object storing value of `ShapeFunction`.

# Examples
```jldoctest
julia> sv = Poingr.ShapeValues{2}(QuadraticBSpline());

julia> update!(sv, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(sv.N)
1.0

julia> sum(sv.∇N)
2-element Vec{2, Float64}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
ShapeValues{dim}(F::ShapeFunction) where {dim} = ShapeValues{dim, Float64}(F)

update!(it::ShapeValues, grid::Grid, x::Vec) = update!(it, grid, x, trues(size(grid)))

function update_gridindices!(it::ShapeValues, grid::Grid{dim}, gridindices::CartesianIndices, spat::AbstractArray{Bool, dim}) where {dim}
    @assert size(grid) == size(spat)
    @boundscheck checkbounds(grid, gridindices)
    count = 0
    @inbounds for I in gridindices
        i = LinearIndices(grid)[I]
        if spat[i]
            @assert count != length(it.inds)
            it.inds[count+=1] = Index(i, I)
        end
    end
    it.len[] = count
    it
end
