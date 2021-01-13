const GridBound{dim} = NamedTuple{(:index, :component, :n), Tuple{CartesianIndex{dim}, Int, Vec{dim, Int}}}

"""
    GridBoundSet(indices::CartesianIndices{dim}, direction::Int)

Create `GridBoundSet` with `indices`.
`i = abs(direction)` represents `i`th component such as `1 ≤ i ≤ dim`.
`sign(direction)` represents the direction of unit normal vector.
The element of `GridBoundSet` has fields `:index`, `:component` and `:n`.

# Examples
```jldoctest
julia> GridBoundSet(CartesianIndices((1:2,1:2)), 2)
GridBoundSet{2} with 4 elements:
  (index = CartesianIndex(1, 1), component = 2, n = [0, 1])
  (index = CartesianIndex(2, 1), component = 2, n = [0, 1])
  (index = CartesianIndex(1, 2), component = 2, n = [0, 1])
  (index = CartesianIndex(2, 2), component = 2, n = [0, 1])
```
"""
struct GridBoundSet{dim} <: AbstractSet{GridBound{dim}}
    set::Set{GridBound{dim}}
end

function GridBoundSet(inds::CartesianIndices{dim}, direction::Int) where {dim}
    n = Vec{dim, Int}(d -> d == abs(direction) ? sign(direction) : 0)
    GridBoundSet(Set([GridBound{dim}((i, abs(direction), n)) for i in inds]))
end

Base.length(bound::GridBoundSet) = length(bound.set)
Base.emptymutable(::GridBoundSet{T}, ::Type{U} = T) where {T, U} = GridBoundSet(Set{U}())

Base.iterate(bound::GridBoundSet, state...) = iterate(bound.set, state...)
Base.push!(bound::GridBoundSet, v) = push!(bound.set, v)
