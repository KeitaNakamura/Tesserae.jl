"""
    GridOffset{N}

An exact displacement in an `N`-dimensional logical grid. `doubled` stores twice
the displacement in each coordinate as an integer. Offsets are normally created
with [`unitoffsets`](@ref) and its arithmetic operations.
"""
struct GridOffset{N}
    doubled::NTuple{N, Int}
end

@inline nhalfsteps(offset::GridOffset, d::Int) = offset.doubled[d]

"""
    unitoffsets(Val(N))

Return the `N` standard basis offsets of the logical grid coordinate space.
"""
function unitoffsets(::Val{N}) where {N}
    ntuple(Val(N)) do d
        GridOffset{N}(ntuple(i -> ifelse(i == d, 2, 0), Val(N)))
    end
end

@inline Base.:+(a::GridOffset{N}, b::GridOffset{N}) where {N} = GridOffset{N}(map(+, a.doubled, b.doubled))
@inline Base.:-(a::GridOffset{N}, b::GridOffset{N}) where {N} = GridOffset{N}(map(-, a.doubled, b.doubled))

@inline Base.:+(offset::GridOffset) = offset
@inline Base.:-(offset::GridOffset) = GridOffset(map(-, offset.doubled))

@inline Base.:*(n::Int, offset::GridOffset) = GridOffset(map(d -> n * d, offset.doubled))
@inline Base.:*(offset::GridOffset, n::Int) = n * offset

function Base.:/(offset::GridOffset, n::Int)
    iszero(n) && throw(DivideError())
    all(d -> iszero(rem(d, n)), offset.doubled) || throw(ArgumentError("result is not representable in half-grid coordinates"))
    GridOffset(map(d -> div(d, n), offset.doubled))
end

Base.zero(::Type{GridOffset{N}}) where {N} = GridOffset(ntuple(_ -> 0, Val(N)))
Base.zero(::GridOffset{N}) where {N} = zero(GridOffset{N})
Base.iszero(offset::GridOffset) = all(iszero, offset.doubled)

function Base.show(io::IO, offset::GridOffset{N}) where {N}
    print(io, "GridOffset(")
    for d in 1:N
        d == 1 || print(io, ", ")
        show(io, _logical_coordinate(offset.doubled[d]))
    end
    print(io, ')')
end

@inline _logical_coordinate(x::Int) = iseven(x) ? x ÷ 2 : x // 2

Base.broadcastable(offset::GridOffset) = Ref(offset)
