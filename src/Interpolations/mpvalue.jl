abstract type Interpolation end
abstract type Kernel <: Interpolation end

Broadcast.broadcastable(interp::Interpolation) = (interp,)

abstract type MPValue{dim, T} end

num_nodes(mp::MPValue) = length(mp.nodeindices)

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
update!(mp::MPValue, grid::Grid, pt) = update!(mp, grid, trues(size(grid)), pt)
function update!(mp::MPValue, grid::Grid, sppat::AbstractArray{Bool}, pt)
    @assert size(grid) == size(sppat)
    mp.nodeindices = nodeindices(get_kernel(mp), grid, pt)
    update_kernels!(mp, grid, sppat, pt)
    mp
end

update_kernels!(mp::MPValue, grid::Grid, sppat, pt) = update_kernels!(mp, grid, sppat, pt.x)
