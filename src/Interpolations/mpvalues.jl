abstract type Kernel <: Interpolation end

abstract type MPValue end
abstract type MPValues{dim, T, V <: MPValue} <: AbstractVector{V} end

Base.size(x::MPValues) = (x.len,)
gridindices(x::MPValues) = x.gridindices
gridindices(x::MPValues, i::Int) = (@_propagate_inbounds_meta; x.gridindices[i])

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

update!(mpvalues::MPValues, grid::Grid, pt, spat::AbstractArray{Bool}) = update!(mpvalues, grid, pt.x, spat)
update!(mpvalues::MPValues, grid::Grid, pt) = update!(mpvalues, grid, pt, trues(size(grid)))

function update_active_gridindices!(mpvalues::MPValues, gridindices::CartesianIndices{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    count = 0
    @inbounds for I in gridindices
        i = LinearIndices(spat)[I]
        if spat[i]
            @assert count != length(mpvalues.gridindices)
            mpvalues.gridindices[count+=1] = Index(i, I)
        end
    end
    mpvalues.len = count
    allactive = count == length(gridindices) == length(mpvalues.N)
    allactive
end
