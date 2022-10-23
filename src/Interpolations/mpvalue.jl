abstract type Interpolation end
abstract type Kernel <: Interpolation end

abstract type MPValue{dim, T} end

num_nodes(mp::MPValue) = mp.len
nodeindex(mp::MPValue, i::Int) = (@_propagate_inbounds_meta; mp.nodeindices[i])

"""
    MPValue{dim}(::Interpolation)
    MPValue{dim, T}(::Interpolation)

Construct object storing value of `Interpolation`.

# Examples
```jldoctest
julia> mp = MPValue{2}(QuadraticBSpline());

julia> update!(mp, Grid(0.0:3.0, 0.0:3.0), Vec(1, 1));

julia> sum(mp.N)
1.0

julia> sum(mp.∇N)
2-element Vec{2, Float64}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
MPValue{dim}(F::Interpolation) where {dim} = MPValue{dim, Float64}(F)

@inline getx(x::Vec) = x
@inline getx(pt) = pt.x
update!(mp::MPValue, grid::Grid, pt) = update!(mp, grid, pt, trues(size(grid)))
function update!(mp::MPValue, grid::Grid, pt, sppat::AbstractArray{Bool})
    @assert size(grid) == size(sppat)
    mp.xp = getx(pt)
    ci = nodeindices(get_kernel(mp), grid, pt)
    update_nodeindices!(mp, ci, sppat)
    update_kernels!(mp, grid, pt)
    mp
end

update_kernels!(mp::MPValue, grid::Grid, pt) = update_kernels!(mp, grid, pt.x)

# filtering by sppat
function update_nodeindices!(mp::MPValue, nodeindices::CartesianIndices{dim}, sppat::AbstractArray{Bool, dim}) where {dim}
    count = 0
    @inbounds for I in nodeindices
        i = LinearIndices(sppat)[I]
        if sppat[i]
            @assert count != length(mp.nodeindices)
            mp.nodeindices[count+=1] = Index(i, I)
        end
    end
    mp.len = count
    mp
end
