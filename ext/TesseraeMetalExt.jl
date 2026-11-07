module TesseraeMetalExt

using Tesserae
using Tesserae: MetalDevice, EltypePolicy, CastFloat32, PreserveEltype

using Metal

using KernelAbstractions
using Adapt
using SparseArrays: AbstractSparseArray

KernelAbstractions.get_backend(x::MetalDevice) = MetalBackend()
Tesserae.get_device(x::MetalBackend) = MetalDevice{EltypePolicy}()
Tesserae.has_device(x::MetalDevice) = true

function check_metal_storage(A)
    A isa AbstractSparseArray &&
        throw(ArgumentError("Metal does not support sparse array storage; transferring it would create a dense array. Sparse matrix assembly on GPU requires CUDA."))
    nothing
end

function Adapt.adapt_storage(::MetalDevice{CastFloat32}, A::AbstractArray)
    check_metal_storage(A)
    mtl(A)
end

function Adapt.adapt_storage(::MetalDevice{PreserveEltype}, A::AbstractArray)
    check_metal_storage(A)
    adapt(MtlArray, A) # DefaultStorageMode
end

function Adapt.adapt_storage(::MetalDevice{CastFloat32}, A::AbstractArray{<: Tensor{S,T,N,L}}) where {S,T<:AbstractFloat,N,L}
    check_metal_storage(A)
    adapt(MtlArray{Tensor{S,Float32,N,L}}, A) # DefaultStorageMode
end

Tesserae.free_temporary!(a::MtlArray) = Metal.unsafe_free!(a)

end
