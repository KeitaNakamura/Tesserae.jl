###############
# HybridArray #
###############

# HybridArray is used to handle atomic operations on the GPU.
# Since atomic operations do not support custom bitstypes such as `Tensor`,
# the data is flattened into a HybridArray.

struct HybridArray{T, N, A <: AbstractArray{T, N}, B <: AbstractArray, D <: AbstractDevice} <: AbstractArray{T, N}
    parent::A
    flat::B
    device::D # stored in advance to avoid the overhead of calling `get_device` in a loop.
end

Base.parent(A::HybridArray) = A.parent
flatten(A::HybridArray) =  A.flat
get_device(A::HybridArray) = A.device

Base.size(A::HybridArray) = size(parent(A))
Base.IndexStyle(::Type{<: HybridArray{<: Any, <: Any, A}}) where {A} = IndexStyle(A)

@inline function Base.getindex(A::HybridArray, I...)
    @boundscheck checkbounds(parent(A), I...)
    @inbounds parent(A)[I...]
end
@inline function Base.setindex!(A::HybridArray, v, I...)
    @boundscheck checkbounds(parent(A), I...)
    @inbounds parent(A)[I...] = v
    A
end

@inline add!(A::AbstractArray{T}, i, v::T) where {T} = (@_propagate_inbounds_meta; A[i] += v)
@inline add!(A::HybridArray{T}, i, v::T) where {T} = (@_propagate_inbounds_meta; _add!(get_device(A), A, i, v))
@inline _add!(::CPUDevice, A::HybridArray, i, v) = (@_propagate_inbounds_meta; add!(parent(A), i, v))

# SpArray atomics write to compact storage through `SpIndex`; dense arrays can
# use their logical index directly.
@inline _atomic_array(A::HybridArray) = parent(A)
@inline _atomic_array(A::HybridArray{<:Any, <:Any, <:SpArray}) = get_data(parent(A))
@inline _atomic_index(A::HybridArray, i) = i
@inline _atomic_index(A::HybridArray{<:Any, <:Any, <:SpArray}, i::SpIndex) = storageindex(i)
@inline function _add!(::GPUDevice, A::HybridArray, i, v::Number)
    @_propagate_inbounds_meta
    Atomix.@atomic _atomic_array(A)[_atomic_index(A, i)] += v
end
@inline function _add!(::GPUDevice, A::HybridArray, i, v::Union{Tensor, StaticArray})
    @_propagate_inbounds_meta
    data = Tuple(v)
    si = _atomic_index(A, i)
    for j in eachindex(data)
        Atomix.@atomic flatten(A)[j,si] += data[j]
    end
end

flatten(A::AbstractArray{T}) where {T <: Number} = reshape(A, 1, size(A)...)
flatten(A::AbstractArray{T}) where {T <: Tensor} = reinterpret(reshape, eltype(T), A)
flatten(A::SpArray{T}) where {T <: Number} = reshape(get_data(A), 1, length(get_data(A)))
flatten(A::SpArray{T}) where {T <: Tensor} = reinterpret(reshape, eltype(T), get_data(A))

get_spinds(A::HybridArray{<:Any, <:Any, <:SpArray}) = get_spinds(parent(A))
get_data(A::HybridArray{<:Any, <:Any, <:SpArray}) = get_data(parent(A))

hybrid(A::AbstractArray{T}) where {T} = HybridArray(A, flatten(A), get_device(A))
hybrid(A::StructArray) = StructArray(map(hybrid, StructArrays.components(A)))
hybrid(mesh::AbstractMesh) = mesh
