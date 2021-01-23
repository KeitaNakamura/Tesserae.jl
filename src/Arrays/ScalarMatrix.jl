struct ScalarMatrix{T} <: AbstractArray{T, 2}
    val::T
    dim::Int
end

Base.size(x::ScalarMatrix) = (x.dim, x.dim)
value(x::ScalarMatrix) = x.val

@inline function Base.getindex(x::ScalarMatrix, i::Int, j::Int)
    @boundscheck checkbounds(x, i, j)
    val = value(x)
    i == j ? val : zero(val)
end
