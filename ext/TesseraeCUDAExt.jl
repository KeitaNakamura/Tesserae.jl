module TesseraeCUDAExt

using Tesserae
using Tesserae: CUDADevice

using CUDA
using CUDA: default_memory

using KernelAbstractions
using Adapt

KernelAbstractions.get_backend(x::CUDADevice) = CUDABackend()
Tesserae.get_device(x::CUDABackend) = CUDADevice()
Tesserae.has_device(x::CUDADevice) = true

function Adapt.adapt_storage(::CUDADevice, A::AbstractArray)
    adapt(CUDA.CuArrayKernelAdaptor{default_memory}(), A)
end

function Adapt.adapt_storage(::CUDADevice, A::AbstractArray{<: Tensor{S,T,N,L}, M}) where {S,T,N,L,M}
    adapt(CuArray{Tensor{S,T,N,L},M,default_memory}, A)
end

function Adapt.adapt_storage(::CUDADevice, A::AbstractArray{<: Tensor{S,T,N,L}, M}) where {S,T<:AbstractFloat,N,L,M}
    adapt(CuArray{Tensor{S,Float32,N,L},M,default_memory}, A)
end

end
