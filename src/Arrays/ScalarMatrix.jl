struct ScalarMatrix{T} <: AbstractArray{T, 2}
    val::T
    dims::NTuple{2, Int}
end
ScalarMatrix(v, i::Int, j::Int) = ScalarMatrix(v, (i,j))

Base.size(x::ScalarMatrix) = x.dims
value(x::ScalarMatrix) = x.val

@inline function Base.getindex(x::ScalarMatrix, i::Int, j::Int)
    @boundscheck checkbounds(x, i, j)
    val = value(x)
    i == j ? val : zero(val)
end
