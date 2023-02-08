abstract type Interpolation end
abstract type Kernel <: Interpolation end

get_kernel(k::Kernel) = k

Broadcast.broadcastable(interp::Interpolation) = (interp,)
@inline neighbornodes(interp::Interpolation, lattice::Lattice, pt) = neighbornodes(get_kernel(interp), lattice, pt)

abstract type MPValue{dim, T, I <: Interpolation} end

MPValue{dim, T, I}() where {dim, T, I} = MPValue{dim, T}(I())

get_interp(::MPValue{<: Any, <: Any, I}) where {I} = I()
get_kernel(mp::MPValue) = get_kernel(get_interp(mp))
num_nodes(mp::MPValue) = length(mp.N)
@inline neighbornodes(mp::MPValue, lattice::Lattice, pt) = neighbornodes(get_interp(mp), lattice, pt)

struct NearBoundary{true_or_false} end

"""
    MPValue{dim}(::Interpolation)
    MPValue{dim, T}(::Interpolation)

Construct object storing value of `Interpolation`.

# Examples
```jldoctest
julia> mp = MPValue{2}(QuadraticBSpline());

julia> update!(mp, Lattice(0.0:3.0, 0.0:3.0), Vec(1, 1));

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
function update!(mp::MPValue, lattice::Lattice, pt)
    update!(mp, lattice, trues(size(lattice)), pt)
end
function update!(mp::MPValue, lattice::Lattice, sppat::Union{AllTrue, AbstractArray{Bool}}, pt)
    update!(mp, lattice, sppat, CartesianIndices(lattice), pt)
end
function update!(mp::MPValue, lattice::Lattice, nodeinds::CartesianIndices, pt)
    update!(mp, lattice, trues(size(lattice)), nodeinds, pt)
end
@inline function update!(mp::MPValue{dim}, lattice::Lattice, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::CartesianIndices, pt) where {dim}
    sppat isa AbstractArray && @assert size(lattice) == size(sppat)
    @boundscheck checkbounds(lattice, nodeinds)
    n = length(nodeinds)
    if n == maxnum_nodes(get_kernel(mp), Val(dim)) && (sppat isa AllTrue || _all(sppat, nodeinds))
        update!(mp, NearBoundary{false}(), lattice, AllTrue(), nodeinds, pt)
    else
        update!(mp, NearBoundary{true}(), lattice, sppat, nodeinds, pt)
    end
    mp
end
# don't check bounds
@inline function _all(A::AbstractArray, inds::CartesianIndices)
    @inbounds @simd for i in inds
        A[i] || return false
    end
    true
end

update!(mp::MPValue, nb::NearBoundary, lattice::Lattice, sppat::Union{AllTrue, AbstractArray{Bool}}, nodeinds::CartesianIndices, pt) = update!(mp, nb, lattice, sppat, nodeinds, pt.x)
