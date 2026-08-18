module TesseraeCUDAExt

using Tesserae
using Tesserae: CUDADevice, EltypePolicy, CastFloat32, PreserveEltype

using CUDA

using KernelAbstractions
using Adapt

KernelAbstractions.get_backend(x::CUDADevice) = CUDABackend()
Tesserae.get_device(x::CUDABackend) = CUDADevice{EltypePolicy}()
Tesserae.has_device(x::CUDADevice) = true

# The same query CUDA's KernelAbstractions backend runs for dynamically sized
# kernels: compile without launching and let the occupancy API size the
# workgroup for this kernel on this device. The probe ndrange only shapes the
# iteration space; the compiled function is independent of it.
function Tesserae.optimal_workgroupsize(::CUDABackend, kernel::KernelAbstractions.Kernel{CUDABackend}, args::Tuple)
    ndrange = (Tesserae.P2G_BLOCK_GROUPSIZE,)
    iterspace, _ = KernelAbstractions.partition(kernel, ndrange, ndrange)
    ctx = KernelAbstractions.mkcontext(kernel, ndrange, iterspace)
    hk = CUDA.@cuda launch=false kernel.f(ctx, args...)
    CUDA.launch_configuration(hk.fun).threads
end

function Adapt.adapt_storage(::CUDADevice{CastFloat32}, A::AbstractArray)
    cu(A)
end

function Adapt.adapt_storage(::CUDADevice{PreserveEltype}, A::AbstractArray)
    adapt(CuArray, A) # default_memory
end

function Adapt.adapt_storage(::CUDADevice{CastFloat32}, A::AbstractArray{<: Tensor{S,T,N,L}}) where {S,T<:AbstractFloat,N,L}
    adapt(CuArray{Tensor{S,Float32,N,L}}, A) # default_memory
end

Tesserae.free_temporary!(a::CuArray) = CUDA.unsafe_free!(a)

end
