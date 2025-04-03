abstract type AbstractDevice end
struct CPUDevice <: AbstractDevice end
abstract type GPUDevice <: AbstractDevice end
struct CUDADevice   <: GPUDevice end
struct AMDGPUDevice <: GPUDevice end
struct MetalDevice  <: GPUDevice end
struct oneAPIDevice <: GPUDevice end

(to::AbstractDevice)(A) = adapt(to, A)
get_device(A) = get_device(get_backend(A))

# extended in extension packages
KernelAbstractions.get_backend(::CPUDevice) = CPU()
get_device(::CPU) = CPUDevice()
has_device(::AbstractDevice) = false

function gpu_device()
    has_device(CUDADevice())   && return CUDADevice()
    has_device(AMDGPUDevice()) && return AMDGPUDevice()
    has_device(MetalDevice())  && return MetalDevice()
    has_device(oneAPIDevice()) && return oneAPIDevice()
    error("No GPU device found")
end
