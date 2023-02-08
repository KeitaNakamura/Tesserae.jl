########
# Grid #
########

const Grid{T, N, C, I} = StructArray{T, N, C, I} where {T, N, C <: NamedTuple{<: Any, <: Tuple{Lattice, Vararg{AbstractArray}}}, I}

generate_grid(::Type{GridState}, dx::Real, minmax::Tuple{Real, Real}...) where {GridState} = generate_grid(GridState, NormalSystem(), dx, minmax...)

get_lattice(grid::Grid) = grid.x
get_system(grid::Grid) = get_system(get_lattice(grid))
spacing(grid::Grid) = spacing(get_lattice(grid))

##########
# SpGrid #
##########

const SpGrid{T, N, C, I} = StructArray{T, N, C, I} where {T, N, C <: NamedTuple{<: Any, <: Tuple{Lattice, SpArray, Vararg{SpArray}}}, I}

# spgrid
@generated function generate_grid(::Type{GridState}, system::CoordinateSystem, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {GridState, dim}
    fieldname(GridState, 1) == :x || return :(error("generate_grid: first field name must be `:x`"))
    V = fieldtype(GridState, 1)
    V <: Vec{dim} || return :(error("generate_grid: `fieldtype` of `:x` must be `<: Vec{$dim}`"))
    exps = [:(SpArray{$T}(sppat, stamp)) for T in fieldtypes(GridState)[2:end]]
    quote
        lattice = Lattice($(eltype(V)), system, dx, minmax...)
        sppat = SpPattern(size(lattice))
        stamp = Ref(NaN)
        StructArray{GridState}(tuple(lattice, $(exps...)))
    end
end

get_stamp(A::SpGrid) = get_stamp(getproperty(A, 2))
set_stamp!(A::SpGrid, v) = set_stamp!(getproperty(A, 2), v)
get_sppat(A::SpGrid) = get_sppat(getproperty(A, 2))

# fillzero!
fillzero!(A::SpGrid) = (StructArrays.foreachfield(_fillzero!, A); A)
_fillzero!(x::Lattice) = x
_fillzero!(x::AbstractArray) = fillzero!(x)

# update_sparsity_pattern!
function update_sparsity_pattern!(A::SpGrid, sppat::AbstractArray{Bool})
    @assert size(A) == size(sppat)
    set_stamp!(A, NaN)
    n = update_sparsity_pattern!(get_sppat(A), sppat)
    StructArrays.foreachfield(a->_resize_nonzeros!(a,n), A)
    A
end
_resize_nonzeros!(x::Lattice, n) = x
_resize_nonzeros!(x::SpArray, n) = resize!(nonzeros(x), n)

# unsafe becuase the returned index can be -1 if the SpPattern is not correctly updated
@inline function unsafe_nonzeroindex(A::SpGrid, i)
    @boundscheck checkbounds(A, i)
    @inbounds NonzeroIndex(get_spindices(get_sppat(A))[i])
end

Base.show(io::IO, mime::MIME"text/plain", x::SpGrid) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpGrid) = show(io, ShowSpArray(x))
