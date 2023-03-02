########
# Grid #
########

const Grid{N, T, C <: NamedTuple{<: Any, <: Tuple{Lattice, Vararg{AbstractArray}}}, I} = StructArray{T, N, C, I}

get_sppat(A::Grid) = Trues(size(A))
get_lattice(grid::Grid) = grid.x
spacing(grid::Grid) = spacing(get_lattice(grid))

##########
# SpGrid #
##########

const SpGrid{N, T, C <: NamedTuple{<: Any, <: Tuple{Lattice, SpArray, Vararg{SpArray}}}, I} = StructArray{T, N, C, I}

# spgrid
@generated function generate_grid(::Type{GridState}, dx::Real, minmax::Vararg{Tuple{Real, Real}, dim}) where {GridState, dim}
    fieldname(GridState, 1) == :x || return :(error("generate_grid: first field name must be `:x`"))
    V = fieldtype(GridState, 1)
    V <: Vec{dim} || return :(error("generate_grid: `fieldtype` of `:x` must be `<: Vec{$dim}`"))
    exps = [:(SpArray{$T}(sppat)) for T in fieldtypes(GridState)[2:end]]
    quote
        lattice = Lattice($(eltype(V)), dx, minmax...)
        sppat = SpPattern(size(lattice))
        StructArray{GridState}(tuple(lattice, $(exps...)))
    end
end
function generate_grid(::Type{T}, dx::Real, minmax::Vararg{Tuple{Real, Real}}) where {T <: Real}
    lattice = Lattice(T, dx, minmax...)
    StructArray((; x = lattice))
end
generate_grid(dx::Real, minmax::Vararg{Tuple{Real, Real}}) = generate_grid(Float64, dx, minmax...)

get_sppat(A::SpGrid) = get_sppat(getproperty(A, 2))

# fillzero!
fillzero!(A::SpGrid) = (StructArrays.foreachfield(_fillzero!, A); A)
_fillzero!(x::Lattice) = x
_fillzero!(x::AbstractArray) = fillzero!(x)

# DON'T manually call this function
# this should be called from `update!` in MPSpace
function unsafe_update_sparsity_pattern!(A::SpGrid, sppat::AbstractArray{Bool})
    @assert size(A) == size(sppat)
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
