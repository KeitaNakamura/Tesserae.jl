module TesseraeMetalExt

using Tesserae
using Tesserae: MetalDevice, EltypePolicy, CastFloat32, PreserveEltype

using Metal
using Metal: DefaultStorageMode

using KernelAbstractions
using Adapt

KernelAbstractions.get_backend(x::MetalDevice) = MetalBackend()
Tesserae.get_device(x::MetalBackend) = MetalDevice{EltypePolicy}()
Tesserae.has_device(x::MetalDevice) = true

function Adapt.adapt_storage(::MetalDevice{CastFloat32}, A::AbstractArray)
    adapt(Metal.MtlArrayAdaptor{DefaultStorageMode}(), A)
end

function Adapt.adapt_storage(::MetalDevice{PreserveEltype}, A::AbstractArray)
    adapt(MtlArray, A) # DefaultStorageMode
end

function Adapt.adapt_storage(::MetalDevice{CastFloat32}, A::AbstractArray{<: Tensor{S,T,N,L}}) where {S,T<:AbstractFloat,N,L}
    adapt(MtlArray{Tensor{S,Float32,N,L}}, A) # DefaultStorageMode
end

end
