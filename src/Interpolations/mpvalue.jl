abstract type Interpolation end
abstract type Kernel <: Interpolation end

get_kernel(k::Kernel) = k

Broadcast.broadcastable(interp::Interpolation) = (interp,)
@inline neighbornodes(interp::Interpolation, grid::Grid, pt) = neighbornodes(get_kernel(interp), grid, pt)

abstract type MPValue{dim, T, I <: Interpolation} end

MPValue{dim, T, I}() where {dim, T, I} = MPValue{dim, T}(I())

get_interp(::MPValue{<: Any, <: Any, I}) where {I} = I()
get_kernel(mp::MPValue) = get_kernel(get_interp(mp))
num_nodes(mp::MPValue) = length(mp.N)
@inline neighbornodes(mp::MPValue, grid::Grid, pt) = neighbornodes(get_interp(mp), grid, pt)

struct NearBoundary{true_or_false} end

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
function update!(mp::MPValue, grid::Grid, nodeinds::CartesianIndices, pt)
    update!(mp, grid, trues(size(grid)), nodeinds, pt)
end
@inline function update!(mp::MPValue{dim}, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::CartesianIndices, pt) where {dim}
    sppat isa AbstractArray && @assert size(grid) == size(sppat)
    @boundscheck checkbounds(grid, nodeinds)
    n = length(nodeinds)
    if n == maxnum_nodes(get_kernel(mp), Val(dim)) && (sppat isa AllTrue || all(@inbounds view(sppat, nodeinds)))
        update!(mp, NearBoundary{false}(), grid, AllTrue(), nodeinds, pt)
    else
        update!(mp, NearBoundary{true}(), grid, sppat, nodeinds, pt)
    end
    mp
end

update!(mp::MPValue, nb::NearBoundary, grid::Grid, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::CartesianIndices, pt) = update!(mp, nb, grid, sppat, nodeinds, pt.x)
