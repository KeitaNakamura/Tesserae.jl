# How a device converts array element types on transfer. `gpu_device` only ever
# builds the two concrete policies, but the backend extensions return
# `CUDADevice{EltypePolicy}()` -- the abstract type substituted as the concrete
# parameter -- to mean "this array is already on that device, policy unknown".
# `GPUDevice{P}`'s subtypes leave `P` unconstrained so that type-checks.
#
# The consequence to keep in mind: `adapt_storage` is defined only for the two
# concrete policies, and type parameters are invariant, so such a device matches
# none of them. It is a tag for identifying the device, never a converter --
# adapting through one silently falls back to the generic `adapt`.
abstract type EltypePolicy end
struct CastFloat32    <: EltypePolicy end
struct PreserveEltype <: EltypePolicy end

abstract type AbstractDevice end
struct CPUDevice <: AbstractDevice end

abstract type GPUDevice{P <: EltypePolicy} <: AbstractDevice end
struct CUDADevice{P}   <: GPUDevice{P} end
struct AMDGPUDevice{P} <: GPUDevice{P} end
struct MetalDevice{P}  <: GPUDevice{P} end
struct oneAPIDevice{P} <: GPUDevice{P} end

(to::AbstractDevice)(A) = adapt(to, A)
get_device(A) = get_device(get_backend(A))

# extended in extension packages
KernelAbstractions.get_backend(::CPUDevice) = CPU()
get_device(::CPU) = CPUDevice()
has_device(::AbstractDevice) = false

function gpu_device(::Type{P}=CastFloat32) where {P<:EltypePolicy}
    has_device(CUDADevice{P}())   && return CUDADevice{P}()
    has_device(AMDGPUDevice{P}()) && return AMDGPUDevice{P}()
    has_device(MetalDevice{P}())  && return MetalDevice{P}()
    has_device(oneAPIDevice{P}()) && return oneAPIDevice{P}()
    error("No GPU device found")
end
