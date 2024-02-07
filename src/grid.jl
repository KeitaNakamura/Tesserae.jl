########
# Grid #
########

const Grid{N, T, NT <: NamedTuple{<: Any, <: Tuple{Lattice, Vararg{AbstractArray}}}, I} = StructArray{T, N, NT, I}

get_lattice(grid::Grid) = getx(grid)
spacing(grid::Grid) = spacing(get_lattice(grid))

# fillzero!
fillzero!(A::Grid) = (StructArrays.foreachfield(_fillzero!, A); A)
_fillzero!(x::Lattice) = x
_fillzero!(x::AbstractArray) = fillzero!(x)

generate_grid(lattice::Lattice) = StructArray((; x=lattice))
generate_grid(dx::Real, minmax::Vararg{Tuple{Real, Real}}) = generate_grid(Lattice(Float64, dx, minmax...))

##########
# SpGrid #
##########

const SpGrid{N, T, NT <: NamedTuple{<: Any, <: Tuple{Lattice, SpArray, Vararg{SpArray}}}, I} = StructArray{T, N, NT, I}

Base.@pure function infer_lattice_realtype(::Type{GridProperty}, ::Val{dim}) where {GridProperty, dim}
    V = fieldtype(GridProperty, 1)
    V <: Vec{dim} || error("generate_grid: the first `fieldtype` must be `<: Vec{$dim}`")
    eltype(V)
end

"""
    generate_grid(Δx::Real, (xmin, xmax)::Tuple{Real, Real}...)
    generate_grid(GridProperty, Δx::Real, (xmin, xmax)::Tuple{Real, Real}...)

Generate background grid with type `GridProperty`.

This returns `StructArray` (see [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl)).
The first field of `GridProperty` must be of type `Vec`.
It is also strongly recommended that `GridProperty` is bits type for performance, i.e., `isbitstype(GridProperty)`
returns `true`.

# Examples
```jldoctest
julia> struct GridProperty{dim, T}
           x::Vec{dim, T}
           m::Float64
           mv::Vec{dim, T}
           f::Vec{dim, T}
           v::Vec{dim, T}
           vⁿ::Vec{dim, T}
       end

julia> grid = generate_grid(GridProperty{2,Float64}, 0.5, (0,3), (0,2));

julia> grid[1]
GridProperty{2, Float64}([0.0, 0.0], 0.0, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

julia> grid.x
7×5 Lattice{2, Float64, Vector{Float64}}:
 [0.0, 0.0]  [0.0, 0.5]  [0.0, 1.0]  [0.0, 1.5]  [0.0, 2.0]
 [0.5, 0.0]  [0.5, 0.5]  [0.5, 1.0]  [0.5, 1.5]  [0.5, 2.0]
 [1.0, 0.0]  [1.0, 0.5]  [1.0, 1.0]  [1.0, 1.5]  [1.0, 2.0]
 [1.5, 0.0]  [1.5, 0.5]  [1.5, 1.0]  [1.5, 1.5]  [1.5, 2.0]
 [2.0, 0.0]  [2.0, 0.5]  [2.0, 1.0]  [2.0, 1.5]  [2.0, 2.0]
 [2.5, 0.0]  [2.5, 0.5]  [2.5, 1.0]  [2.5, 1.5]  [2.5, 2.0]
 [3.0, 0.0]  [3.0, 0.5]  [3.0, 1.0]  [3.0, 1.5]  [3.0, 2.0]

julia> grid.v
7×5 Matrix{Vec{2, Float64}}:
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
 [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]  [0.0, 0.0]
```
"""
function generate_grid(::Type{ArrayType}, ::Type{GridProperty}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {ArrayType, GridProperty, dim}
    T = infer_lattice_realtype(GridProperty, Val(dim))
    generate_grid(ArrayType, GridProperty, Lattice(T, dx, minmax...))
end
function generate_grid(::Type{GridProperty}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {GridProperty, dim}
    generate_grid(Array, GridProperty, dx, minmax...)
end

# from lattice
@generated function generate_grid(::Type{Array}, ::Type{GridProperty}, lattice::Lattice{dim, T}) where {GridProperty, dim, T}
    @assert infer_lattice_realtype(GridProperty, Val(dim)) == T
    exps = [:(Array{$T}(undef, size(lattice))) for T in fieldtypes(GridProperty)[2:end]]
    quote
        fillzero!(StructArray{GridProperty}(tuple(lattice, $(exps...))))
    end
end
@generated function generate_grid(::Type{SpArray}, ::Type{GridProperty}, lattice::Lattice{dim, T}) where {GridProperty, dim, T}
    @assert infer_lattice_realtype(GridProperty, Val(dim)) == T
    exps = [:(SpArray{$T}(spinds)) for T in fieldtypes(GridProperty)[2:end]]
    quote
        spinds = SpIndices(size(lattice))
        StructArray{GridProperty}(tuple(lattice, $(exps...)))
    end
end

get_spinds(A::SpGrid) = get_spinds(getproperty(A, 2))

function update_block_sparsity!(A::SpGrid, blkspy)
    n = update_block_sparsity!(get_spinds(A), blkspy)
    StructArrays.foreachfield(a->resize_data!(a,n), A)
    A
end

@inline isactive(A::SpGrid, I...) = (@_propagate_inbounds_meta; isactive(get_spinds(A), I...))

Base.show(io::IO, mime::MIME"text/plain", x::SpGrid) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpGrid) = show(io, ShowSpArray(x))
