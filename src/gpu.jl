###########
# cpu/gpu #
###########

function Adapt.adapt_storage(::CPUDevice, A::AbstractArray)
    get_device(A) isa CPUDevice ? A : Array(A)
end

cpu(A) = A |> CPUDevice()
gpu(A) = A |> gpu_device()

# directly define `gpu` for `StructArray` since `adapt_structure` is already defined in `StructArrays.jl` package
function (gpu::GPUDevice)(A::StructArray)
    named_tuple = map(gpu, StructArrays.components(A))
    StructArray(named_tuple) # always convert to NamedTuple
end

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
    it = getfield(mpvalues, :it)
    prop = map(a -> adapt(to, a), getfield(mpvalues, :prop))
    indices = adapt(to, getfield(mpvalues, :indices))
    It = typeof(it)
    Prop = typeof(prop)
    Indices = typeof(indices)
    ElType = Base._return_type(_getindex, Tuple{It, Prop, Indices, Int})
    MPValueArray{It, Prop, Indices, ElType, N}(it, prop, indices)
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
