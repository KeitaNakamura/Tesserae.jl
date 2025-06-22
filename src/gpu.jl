###########
# cpu/gpu #
###########

# NOTE: `gpu` always tries to convert Float64 to Float32 (is this really good?)

function Adapt.adapt_storage(::CPUDevice, A::AbstractArray)
    get_device(A) isa CPUDevice ? A : Array(A)
end

cpu(A) = A |> CPUDevice()
gpu(A) = A |> gpu_device()

###################################
# Special conversions in Tesserae #
###################################

# Unlike StructArrays.jl, this also `adapt` each array `to` GPU (no need?)
function Adapt.adapt_structure(to::GPUDevice, A::StructArray)
    named_tuple = map(a -> adapt(to, a), StructArrays.components(A))
    StructArray(named_tuple) # always convert to NamedTuple
end

function Adapt.adapt_structure(to::GPUDevice, x::StepRangeLen{T, R, S, L}) where {T, R, S, L}
    Tnew = T <: AbstractFloat ? Float32 : T
    Rnew = (R <: AbstractFloat || R <: Base.TwicePrecision) ? Float32 : R
    Snew = (S <: AbstractFloat || S <: Base.TwicePrecision) ? Float32 : S
    StepRangeLen{Tnew, Rnew, Snew, L}(x)
end

#######################
# GPU compatibilities #
#######################

# CartesianMesh
function Adapt.adapt_structure(to, mesh::CartesianMesh)
    axes = map(a -> adapt(to, a), mesh.axes)
    T = eltype(eltype(axes))
    CartesianMesh(axes, T(spacing(mesh)), T(spacing_inv(mesh)))
end
function KernelAbstractions.get_backend(mesh::CartesianMesh)
    @assert allequal(map(get_backend, mesh.axes))
    get_backend(mesh.axes[1])
end

# UnstructuredMesh
function KernelAbstractions.get_backend(mesh::UnstructuredMesh)
    backend = get_backend(mesh.nodes)
    @assert get_backend(mesh.cellnodeindices) == backend
    backend
end

# MPValueArray
function Adapt.adapt_structure(to, mpvalues::MPValueArray{<: Any, <: Any, <: Any, <: Any, N}) where {N}
    interp = getfield(mpvalues, :interp)
    prop = map(a -> adapt(to, a), getfield(mpvalues, :prop))
    indices = adapt(to, getfield(mpvalues, :indices))
    Interp = typeof(interp)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{Interp, Prop, Indices, Int})
    MPValueArray{Interp, Prop, Indices, ElType, N}(interp, prop, indices)
end
function KernelAbstractions.get_backend(mpvalues::MPValueArray)
    prop = getfield(mpvalues, :prop)
    backend = get_backend(first(values(prop)))
    @assert all(==(backend), map(get_backend, prop))
    @assert get_backend(getfield(mpvalues, :indices)) == backend
    backend
end

# SpIndices
function KernelAbstractions.get_backend(A::SpIndices)
    get_backend(A.blkinds)
end

# SpArray
function KernelAbstractions.get_backend(A::SpArray)
    backend = get_backend(A.data)
    @assert get_backend(A.spinds) == backend
    backend
end

# HybridArray
function Adapt.adapt_structure(to, A::HybridArray)
    HybridArray(adapt(to, parent(A)), adapt(to, flatten(A)), get_device(A))
end
