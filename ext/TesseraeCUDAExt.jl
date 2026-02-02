module TesseraeCUDAExt

using Tesserae
using Tesserae: CUDADevice, EltypePolicy, CastFloat32, PreserveEltype

using CUDA
using CUDA: default_memory

using KernelAbstractions
using Adapt

KernelAbstractions.get_backend(x::CUDADevice) = CUDABackend()
Tesserae.get_device(x::CUDABackend) = CUDADevice{EltypePolicy}()
Tesserae.has_device(x::CUDADevice) = true

function Adapt.adapt_storage(::CUDADevice{CastFloat32}, A::AbstractArray)
    adapt(CUDA.CuArrayKernelAdaptor{default_memory}(), A)
end

function Adapt.adapt_storage(::CUDADevice{PreserveEltype}, A::AbstractArray)
    adapt(CuArray, A) # default_memory
end

function Adapt.adapt_storage(::CUDADevice{CastFloat32}, A::AbstractArray{<: Tensor{S,T,N,L}}) where {S,T<:AbstractFloat,N,L}
    adapt(CuArray{Tensor{S,Float32,N,L}}, A) # default_memory
end

end
