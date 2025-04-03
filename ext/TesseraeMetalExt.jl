module TesseraeMetalExt

using Tesserae
using Tesserae: MetalDevice

using Metal
using Metal: DefaultStorageMode

using KernelAbstractions
using Adapt

KernelAbstractions.get_backend(x::MetalDevice) = MetalBackend()
Tesserae.get_device(x::MetalBackend) = MetalDevice()
Tesserae.has_device(x::MetalDevice) = true

function Adapt.adapt_storage(::MetalDevice, A::AbstractArray)
    adapt(Metal.MtlArrayAdaptor{DefaultStorageMode}(), A)
end

function Adapt.adapt_storage(::MetalDevice, A::AbstractArray{<: Tensor{S,T,N,L}, M}) where {S,T,N,L,M}
    adapt(MtlArray{Tensor{S,T,N,L},M,DefaultStorageMode}, A)
end

function Adapt.adapt_storage(::MetalDevice, A::AbstractArray{<: Tensor{S,T,N,L}, M}) where {S,T<:AbstractFloat,N,L,M}
    adapt(MtlArray{Tensor{S,Float32,N,L},M,DefaultStorageMode}, A)
end

end
