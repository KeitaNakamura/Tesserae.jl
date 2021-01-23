abstract type ListGroup{Group} end

getgroup(::ListGroup{Group}) where {Group} = Group

struct List{Group, L, T <: NTuple{L, ListGroup{Group}}}
    data::T
end
List(data...) = List(data)

getgroup(::List{Group}) where {Group} = Group

Base.Tuple(list::List) = list.data
@inline Base.length(list::List) = length(Tuple(list))
@inline Base.eachindex(list::List) = Base.OneTo(length(list))
@inline Base.getindex(l::List, i::Int) = (@_propagate_inbounds_meta; l.data[i])
@inline Base.iterate(l::List, i = 1) = (i % UInt) - 1 < length(l) ? (@inbounds l[i], i + 1) : nothing

Base.:+(x::ListGroup{Group}, y::ListGroup{Group}) where {Group} = List((x, y))
Base.:+(x::ListGroup{Group}, list::List{Group}) where {Group} = List(x, Tuple(list)...)
Base.:+(list::List{Group}, x::ListGroup{Group}) where {Group} = List(Tuple(list)..., x)
Base.:+(x::List{Group}, y::List{Group}) where {Group} = List(Tuple(x)..., Tuple(y)...)

# wrong methods
## +
group_error(x, y) = throw(ArgumentError("`Group` must be the same in `+` operator, got $(getgroup(x)) and $(getgroup(y))"))
Base.:+(x::ListGroup, y::ListGroup) = group_error(x, y)
Base.:+(x::ListGroup, list::List) = group_error(x, list)
Base.:+(list::List, x::ListGroup) = group_error(list, x)
Base.:+(x::List, y::List) = group_error(x, y)
