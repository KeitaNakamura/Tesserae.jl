"""
    Grid(axes::AbstractVector...)

Construct `Grid` by `axes`.

# Examples
```jldoctest
julia> grid = Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{2,Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{dim, T, V} <: AbstractGrid{dim, T}
    axisarray::Coordinate{dim, T, V}
end

function Grid(axs::NTuple{dim, V}) where {dim, T, V <: AbstractVector{T}}
    Grid{dim, T, V}(Coordinate(axs))
end

Grid(axs::Vararg{AbstractVector}) = Grid(axs)

getaxisarray(grid::Grid) = grid.axisarray
