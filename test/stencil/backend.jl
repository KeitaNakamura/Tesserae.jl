# A backend with its own view type and broadcast style verifies that execution
# is handed back to the parent's numerical implementation after lowering.
struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    calls::Base.RefValue{Int}
end
Base.size(A::TrackedArray) = size(A.data)
function Base.getindex(::TrackedArray, I::Vararg{Int})
    error("scalar indexing is disabled for this backend")
end
function Base.setindex!(::TrackedArray, v, I::Vararg{Int})
    error("scalar indexing is disabled for this backend")
end
function Base.fill!(A::TrackedArray, value)
    fill!(A.data, value)
    A
end
Base.view(A::TrackedArray,I...) = TrackedArray(view(A.data,I...),A.calls)
function Base.similar(A::TrackedArray, ::Type{T}, dims::Dims) where {T}
    TrackedArray(similar(A.data, T, dims), Ref(0))
end
struct TrackedStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
TrackedStyle(::Val{N}) where {N} = TrackedStyle{N}()
TrackedStyle{N}(::Val{M}) where {N,M} = TrackedStyle{M}()
Base.Broadcast.BroadcastStyle(::Type{<:TrackedArray{T,N}}) where {T,N} = TrackedStyle{N}()
untrack(A::TrackedArray) = A.data
untrack(x) = x
untrack(bc::Base.Broadcast.Broadcasted) = Base.Broadcast.broadcasted(bc.f,map(untrack,bc.args)...)
function Base.copyto!(A::TrackedArray,bc::Base.Broadcast.Broadcasted{<:TrackedStyle})
    A.calls[] += 1
    Base.Broadcast.materialize!(A.data,untrack(bc))
    A
end
