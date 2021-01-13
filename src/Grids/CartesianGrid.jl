"""
    CartesianGrid(origin, dx::Real, ncells)
    CartesianGrid(dx::Real, ncells)

Create cartesian-grid with `ncells` and linear space `dx`.
If `origin` is not given, it is set to zero.
Default boundary sets storing `GridBound` are also generated, which are `"-x"`, `"+x"`, `"-y"`, `"+y"`, `"-z"` and `"+z"` for each dimension.

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

julia> all(bd -> bd.n == [1, 0], getboundset(grid, "-x"))
true

julia> grid == CartesianGrid((0, 0), 1.0, 5)
true
```
"""
struct CartesianGrid{dim, T} <: AbstractGrid{dim, T}
    axisarray::CartesianAxisArray{dim, T}
    boundsets::Dict{String, Set{GridBound{dim}}}
end

# from CartesianAxisArray
function CartesianGrid{dim, T}(axisarray::CartesianAxisArray{dim, T}) where {dim, T}
    CartesianGrid{dim, T}(axisarray, Dict())
end
CartesianGrid{dim}(axisarray::CartesianAxisArray{dim, T}) where {dim, T} = CartesianGrid{dim, T}(axisarray)
CartesianGrid(axisarray::CartesianAxisArray{dim, T}) where {dim, T} = CartesianGrid{dim, T}(axisarray)

for params in ((:dim, :T), (:dim,), ())
    @eval begin
        # with origin
        function CartesianGrid{$(params...)}(origin::Union{Real, Tuple}, dx::Real, ncells::Union{Int, Tuple}) where {$(params...)}
            axisarray = CartesianAxisArray{$(params...)}(origin, dx, ncells .+ 1)
            grid = CartesianGrid{$(params...)}(axisarray)
            generate_default_boundsets!(grid)
            grid
        end
        # without origin
        function CartesianGrid{$(params...)}(dx::Real, ncells::Union{Int, Tuple}) where {$(params...)}
            axisarray = CartesianAxisArray{$(params...)}(0, dx, ncells .+ 1)
            grid = CartesianGrid{$(params...)}(axisarray)
            generate_default_boundsets!(grid)
            grid
        end
    end
end

getaxisarray(grid::CartesianGrid) = grid.axisarray
