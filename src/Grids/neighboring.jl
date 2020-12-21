function neighboring_nodes(ax::AbstractVector, x::Real, h::Real)
    xmin = first(ax)
    xmax = last(ax)
    xmin ≤ x ≤ xmax || return 1:0
    ξ = (x - xmin) / step(ax)
    _neighboring_nodes(ξ, h, length(ax))
end

function _neighboring_nodes(ξ::Real, h::Real, len::Int)
    ξ_l = ξ - h
    ξ_r = ξ + h
    l = ceil(ξ_l)
    r = floor(ξ_r)
    l === ξ_l && (l += 1)
    r === ξ_r && (r -= 1)
    start, stop = clamp.((Int(l+1), Int(r+1)), 1, len) # cut violated indices
    start:stop
end

"""
    neighboring_nodes(grid, x::Vec, h::Real)

Return `CartesianIndices` storing neighboring node indices around `x`.
`h` is a range for searching and its unit is `gridsteps` `dx`.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = CartesianGrid(1.0, (5,))
6-element CartesianGrid{1,Float64}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> neighboring_nodes(grid, Vec(1.5), 1)
2-element CartesianIndices{1,Tuple{UnitRange{Int64}}}:
 CartesianIndex(2,)
 CartesianIndex(3,)

julia> neighboring_nodes(grid, Vec(1.5), 2)
4-element CartesianIndices{1,Tuple{UnitRange{Int64}}}:
 CartesianIndex(1,)
 CartesianIndex(2,)
 CartesianIndex(3,)
 CartesianIndex(4,)
```
"""
@generated function neighboring_nodes(grid::AbstractGrid{dim}, x::Vec{dim}, h::Real) where {dim}
    quote
        @_inline_meta
        @inbounds CartesianIndices(
            @ntuple $dim d -> begin
                ax = gridaxes(grid, d)
                neighboring_nodes(ax, x[d], h)
            end
        )
    end
end

"""
    whichcell(grid, x::Vec)

Return cell index where `x` locates.

# Examples
```jldoctest
julia> grid = CartesianGrid(1.0, (5, 5))
6×6 CartesianGrid{2,Float64}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> whichcell(grid, Vec(1.5, 1.5))
CartesianIndex(2, 2)
```
"""
@generated function whichcell(grid::AbstractGrid{dim}, x::Vec{dim}) where {dim}
    quote
        ncells = size(grid) .- 1
        dx = gridsteps(grid)
        xmin = gridorigin(grid)
        xmax = xmin .+ dx .* ncells
        @inbounds begin
            (@nall $dim d -> xmin[d] ≤ x[d] ≤ xmax[d]) || return nothing
            CartesianIndex(@ntuple $dim d -> floor(Int, (x[d] - xmin[d]) / dx[d]) + 1)
        end
    end
end

"""
    neighboring_cells(grid, x::Vec, h::Int)
    neighboring_cells(grid, cellindex::CartesianIndex, h::Int)

Return `CartesianIndices` storing neighboring cell indices around `x`.
`h` is number of outer cells around cell where `x` locates.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = CartesianGrid(1.0, (5, 5))
6×6 CartesianGrid{2,Float64}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> x = Vec(1.5, 1.5);

julia> neighboring_cells(grid, x, 1)
3×3 CartesianIndices{2,Tuple{UnitRange{Int64},UnitRange{Int64}}}:
 CartesianIndex(1, 1)  CartesianIndex(1, 2)  CartesianIndex(1, 3)
 CartesianIndex(2, 1)  CartesianIndex(2, 2)  CartesianIndex(2, 3)
 CartesianIndex(3, 1)  CartesianIndex(3, 2)  CartesianIndex(3, 3)

julia> neighboring_cells(grid, whichcell(grid, x), 1) == ans
true
```
"""
@generated function neighboring_cells(grid::AbstractGrid{dim}, cellindex::CartesianIndex{dim}, h::Int) where {dim}
    quote
        @boundscheck checkbounds(grid, cellindex + oneunit(cellindex))
        @inbounds CartesianIndices(
            @ntuple $dim d -> begin
                i = cellindex[d]
                len = size(grid, d) - 1
                start, stop = clamp.((i-h, i+h), 1, len) # cut violated indices
                start:stop
            end
        )
    end
end

@inline function neighboring_cells(grid::AbstractGrid{dim}, x::Vec{dim}, h::Int) where {dim}
    neighboring_cells(grid, whichcell(grid, x), h)
end
