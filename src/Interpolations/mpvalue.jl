abstract type Interpolation end
abstract type Kernel <: Interpolation end

Broadcast.broadcastable(interp::Interpolation) = (interp,)

abstract type MPValue{dim, T} end

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
@inline getx(pt) = @inbounds pt.x
@inline function alltrue(A::AbstractArray, indices::CartesianIndices)
    @boundscheck checkbounds(A, indices)
    @inbounds @simd for i in indices
        A[i] || return false
    end
    true
end
@inline function alltrue(A::Trues, indices::CartesianIndices)
    @boundscheck checkbounds(A, indices)
    true
end

@inline function update!(mp::MPValue, lattice::Lattice, pt)
    indices = update_mpvalue!(mp, lattice, pt)
    indices isa CartesianIndices || error("`update_mpvalue` must return `CartesianIndices`")
    @assert length(indices) == num_nodes(mp)
    indices
end
@inline function update!(mp::MPValue, lattice::Lattice, sppat::AbstractArray{Bool}, pt)
    @assert size(lattice) == size(sppat)
    indices = update_mpvalue!(mp, lattice, sppat, pt)
    indices isa CartesianIndices || error("`update_mpvalue` must return `CartesianIndices`")
    @assert length(indices) == num_nodes(mp)
    indices
end

@inline function update_mpvalue!(mp::MPValue, lattice::Lattice, pt)
    update_mpvalue!(mp, lattice, Trues(size(lattice)), pt)
end
@inline function update_mpvalue!(mp::MPValue, lattice::Lattice, sppat::AbstractArray, pt)
    sppat isa Trues || @warn "Sparsity pattern on grid is not supported in `$(typeof(mp))`, just ignored" maxlog=1
    update_mpvalue!(mp, lattice, pt)
end
