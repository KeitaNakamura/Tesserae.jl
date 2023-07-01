########
# Grid #
########

const Grid{N, T, NT <: NamedTuple{<: Any, <: Tuple{Lattice, Vararg{AbstractArray}}}, I} = StructArray{T, N, NT, I}

eachnode(grid::Grid) = LazyRows(grid)
get_lattice(grid::Grid) = grid.x
spacing(grid::Grid) = spacing(get_lattice(grid))

# fillzero!
fillzero!(A::Grid) = (StructArrays.foreachfield(_fillzero!, A); A)
_fillzero!(x::Lattice) = x
_fillzero!(x::AbstractArray) = fillzero!(x)

generate_grid(lattice::Lattice) = StructArray((; x = lattice))
generate_grid(dx::Real, minmax::Vararg{Tuple{Real, Real}}) = generate_grid(Lattice(Float64, dx, minmax...))

##########
# SpGrid #
##########

const SpGrid{N, T, NT <: NamedTuple{<: Any, <: Tuple{Lattice, SpArray, Vararg{SpArray}}}, I} = StructArray{T, N, NT, I}

Base.@pure function infer_lattice_realtype(::Type{GridState}, ::Val{dim}) where {GridState, dim}
    fieldname(GridState, 1) == :x || error("generate_grid: first field name must be `:x`")
    V = fieldtype(GridState, 1)
    V <: Vec{dim} || error("generate_grid: `fieldtype` of `:x` must be `<: Vec{$dim}`")
    eltype(V)
end

"""
    generate_grid(Δx::Real, (xmin, xmax)::Tuple{Real, Real}...)
    generate_grid(GridState, Δx::Real, (xmin, xmax)::Tuple{Real, Real}...)

Generate background grid with type `GridState`.

This returns `StructArray` (see [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl)).
`GridState` must have `x::Vec` as its first field.
It is also strongly recommended that `GridState` is bits type, i.e., `isbitstype(GridState)`
returns `true`. It is possible to use `NamedTuple` for `GridState`.
If `GridState` is not given, field `x` is only created as field.

# Examples
```jldoctest
julia> struct GridState{dim, T}
           x::Vec{dim, T}
           m::Float64
           mv::Vec{dim, T}
           f::Vec{dim, T}
           v::Vec{dim, T}
           vⁿ::Vec{dim, T}
       end

julia> grid = generate_grid(GridState{2,Float64}, 0.5, (0,3), (0,2))
7×5 StructArray(::Lattice{2, Float64, Marble.LinAxis{Float64}}, ::Marble.SpArray{Float64, 2, Vector{Float64}, Matrix{UInt32}}, ::Marble.SpArray{Vec{2, Float64}, 2, Vector{Vec{2, Float64}}, Matrix{UInt32}}, ::Marble.SpArray{Vec{2, Float64}, 2, Vector{Vec{2, Float64}}, Matrix{UInt32}}, ::Marble.SpArray{Vec{2, Float64}, 2, Vector{Vec{2, Float64}}, Matrix{UInt32}}, ::Marble.SpArray{Vec{2, Float64}, 2, Vector{Vec{2, Float64}}, Matrix{UInt32}}) with eltype GridState{2, Float64}:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> grid[1]
GridState{2, Float64}([0.0, 0.0], 0.0, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

julia> grid.x
7×5 Lattice{2, Float64, Marble.LinAxis{Float64}}:
 [0.0, 0.0]  [0.0, 0.5]  [0.0, 1.0]  [0.0, 1.5]  [0.0, 2.0]
 [0.5, 0.0]  [0.5, 0.5]  [0.5, 1.0]  [0.5, 1.5]  [0.5, 2.0]
 [1.0, 0.0]  [1.0, 0.5]  [1.0, 1.0]  [1.0, 1.5]  [1.0, 2.0]
 [1.5, 0.0]  [1.5, 0.5]  [1.5, 1.0]  [1.5, 1.5]  [1.5, 2.0]
 [2.0, 0.0]  [2.0, 0.5]  [2.0, 1.0]  [2.0, 1.5]  [2.0, 2.0]
 [2.5, 0.0]  [2.5, 0.5]  [2.5, 1.0]  [2.5, 1.5]  [2.5, 2.0]
 [3.0, 0.0]  [3.0, 0.5]  [3.0, 1.0]  [3.0, 1.5]  [3.0, 2.0]

julia> grid.v
7×5 Marble.SpArray{Vec{2, Float64}, 2, Vector{Vec{2, Float64}}, Matrix{UInt32}}:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅
```
"""
function generate_grid(::Type{GridState}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {GridState, dim}
    T = infer_lattice_realtype(GridState, Val(dim))
    generate_grid(GridState, Lattice(T, dx, minmax...))
end
# from lattice
@generated function generate_grid(::Type{GridState}, lattice::Lattice{dim, T}) where {GridState, dim, T}
    @assert infer_lattice_realtype(GridState, Val(dim)) == T
    exps = [:(SpArray{$T}(spinds)) for T in fieldtypes(GridState)[2:end]]
    quote
        spinds = SpIndices(size(lattice))
        StructArray{GridState}(tuple(lattice, $(exps...)))
    end
end

get_spinds(A::SpGrid) = get_spinds(getproperty(A, 2))

function update_sparsity!(A::SpGrid, blkspace::AbstractArray{Bool})
    n = update_sparsity!(get_spinds(A), blkspace)
    StructArrays.foreachfield(a->resize_nonzeros!(a,n), A)
    A
end

blocksparsity(A::SpGrid) = blocksparsity(get_spinds(A))

@inline isnonzero(A::SpGrid, I...) = (@_propagate_inbounds_meta; isnonzero(get_spinds(A), I...))


@inline function nonzeroindex(A::SpGrid, i)
    @_propagate_inbounds_meta
    nonzeroindex(get_spinds(A), i)
end

Base.show(io::IO, mime::MIME"text/plain", x::SpGrid) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpGrid) = show(io, ShowSpArray(x))
