abstract type Interpolation end
abstract type Kernel <: Interpolation end

Broadcast.broadcastable(interp::Interpolation) = (interp,)
neighbornodes(interp::Interpolation, grid::Grid, pt) = neighbornodes(get_kernel(interp), grid, pt)

abstract type MPValue{dim, T} end

num_nodes(mp::MPValue) = length(mp.N)
neighbornodes(mp::MPValue, grid::Grid, pt) = neighbornodes(get_kernel(mp), grid, pt)

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
function update!(mp::MPValue, grid::Grid, pt)
    update!(mp, grid, trues(size(grid)), pt)
end
function update!(mp::MPValue, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, pt)
    update!(mp, grid, sppat, CartesianIndices(grid), pt)
end
function update!(mp::MPValue, grid::Grid, inds::AbstractArray, pt)
    update!(mp, grid, trues(size(grid)), inds, pt)
end
function update!(mp::MPValue, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, inds::AbstractArray, pt)
    sppat isa AbstractArray && @assert size(grid) == size(sppat)
    @boundscheck checkbounds(grid, inds)
    update_kernels!(mp, grid, sppat, inds, pt)
    mp
end

update_kernels!(mp::MPValue, grid::Grid, sppat, inds, pt) = update_kernels!(mp, grid, sppat, inds, pt.x)
