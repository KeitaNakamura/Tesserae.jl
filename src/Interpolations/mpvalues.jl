abstract type Kernel <: Interpolation end

abstract type MPValues{dim, T} <: AbstractVector{T} end
abstract type MPValue end

Base.size(x::MPValues) = (x.len,)

"""
    MPValues{dim}(::Interpolation)
    MPValues{dim, T}(::Interpolation)

Construct object storing value of `Interpolation`.

# Examples
```jldoctest
julia> mpvalues = MPValues{2}(QuadraticBSpline());

julia> update!(mpvalues, Grid(0.0:3.0, 0.0:3.0), Vec(1, 1));

julia> sum(mpvalues.N)
1.0

julia> sum(mpvalues.∇N)
2-element Vec{2, Float64}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
MPValues{dim}(F::Interpolation) where {dim} = MPValues{dim, Float64}(F)

update!(mpvalues::MPValues, grid::Grid, x::Vec) = update!(mpvalues, grid, x, trues(size(grid)))
update!(mpvalues::MPValues, grid::Grid, x::Vec, r::Vec) = update!(mpvalues, grid, x, r, trues(size(grid)))

function update_gridindices!(mpvalues::MPValues, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    @assert size(grid) == size(spat)
    gridindices = neighboring_nodes(grid, x, support_length(mpvalues.F))
    count = 0
    @inbounds for I in gridindices
        i = LinearIndices(grid)[I]
        if spat[i]
            @assert count != length(mpvalues.gridindices)
            mpvalues.gridindices[count+=1] = Index(i, I)
        end
    end
    mpvalues.len = count
    count == length(gridindices) == length(mpvalues.N)
end
