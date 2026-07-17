###########
# cpu/gpu #
###########

# NOTE: `gpu` always tries to convert Float64 to Float32 (is this really good?)

function Adapt.adapt_storage(::CPUDevice, A::AbstractArray)
    get_device(A) isa CPUDevice ? A : Array(A)
end

# These helpers call `adapt` with a Tesserae `AbstractDevice` as `to`.
# Methods specialized on `to::AbstractDevice` are therefore explicit Tesserae
# device transfers, while unspecialized `adapt_structure(to, ...)` methods may
# also be used by other Adapt callers.
cpu(A) = A |> CPUDevice()
gpu(A) = A |> gpu_device(CastFloat32)
gpu_preserve(A) = A |> gpu_device(PreserveEltype)

###################################
# Special conversions in Tesserae #
###################################

# Unlike StructArrays.jl, this also `adapt` each array `to` GPU (no need?)
function Adapt.adapt_structure(to::GPUDevice, A::StructArray)
    named_tuple = map(a -> adapt(to, a), StructArrays.components(A))
    StructArray(named_tuple) # always convert to NamedTuple
end

_spgrid_sparray(A::SpArray) = A
_spgrid_sparray(A::HybridArray{<:Any, <:Any, <:SpArray}) = parent(A)

function Adapt.adapt_structure(to::GPUDevice, A::SpGrid)
    components = StructArrays.components(A)
    names = propertynames(components)
    mesh = adapt(to, get_mesh(A))
    spinds = adapt(to, get_spinds(A))
    arrays = map(Base.tail(names)) do name
        a = _spgrid_sparray(getproperty(components, name))
        SpArray(adapt(to, get_data(a)), spinds, a.shared_spinds)
    end
    StructArray(NamedTuple{names}((mesh, arrays...)))
end

function Adapt.adapt_structure(to::GPUDevice{CastFloat32}, x::StepRangeLen{T, R, S, L}) where {T, R, S, L}
    Tnew = T <: AbstractFloat ? Float32 : T
    Rnew = (R <: AbstractFloat || R <: Base.TwicePrecision) ? Float32 : R
    Snew = (S <: AbstractFloat || S <: Base.TwicePrecision) ? Float32 : S
    StepRangeLen{Tnew, Rnew, Snew, L}(x)
end

#######################
# GPU compatibilities #
#######################

KernelAbstractions.get_backend(::BitArray) = CPU() # should be implemented in KernelAbstractions.jl

# CartesianMesh
function Adapt.adapt_structure(to, mesh::CartesianMesh)
    axes = map(a -> adapt(to, a), mesh.axes)
    T = eltype(eltype(axes))
    CartesianMesh(axes, T(spacing(mesh)), T(spacing_inv(mesh)); block_size_log2=Val(block_size_log2(mesh)))
end
function KernelAbstractions.get_backend(mesh::CartesianMesh)
    @assert allequal(map(get_backend, mesh.axes))
    get_backend(mesh.axes[1])
end

# FEMesh
function KernelAbstractions.get_backend(mesh::FEMesh)
    backend = get_backend(mesh.nodes)
    @assert get_backend(cellsupports(mesh)) == backend
    @assert get_backend(supportnodes(mesh)) == backend
    backend
end

# IGAMesh
KernelAbstractions.get_backend(mesh::IGAMesh) = get_backend(mesh.controlpoints)

# QuadraturePoints
Adapt.adapt_structure(to, points::QuadraturePoints) = QuadraturePoints(adapt(to, parent(points)), quadrature_rule(points))
KernelAbstractions.get_backend(points::QuadraturePoints) = get_backend(parent(points))

# BasisWeightArray
Adapt.adapt_structure(to, A::CellSupportMatrix) = CellSupportMatrix(adapt(to, cellsupports(A)), size(A)...)
KernelAbstractions.get_backend(A::CellSupportMatrix) = get_backend(cellsupports(A))

function Adapt.adapt_structure(to, weights::BasisWeightArray)
    b = basis(weights)
    vals = map(a -> adapt(to, a), getfield(weights, :vals))
    indices = adapt(to, getfield(weights, :indices))
    BasisWeightArray(b, vals, indices)
end
function KernelAbstractions.get_backend(weights::BasisWeightArray)
    vals = getfield(weights, :vals)
    backend = get_backend(first(values(vals)))
    @assert all(==(backend), map(get_backend, vals))
    @assert get_backend(getfield(weights, :indices)) == backend
    backend
end

# SpIndices
function Adapt.adapt_structure(to::AbstractDevice, A::SpIndices{dim, L}) where {dim, L}
    numbers = adapt(to, blocknumbering(A))
    workspace = BlockSparsityWorkspace(numbers)
    SpIndices{dim, L, typeof(numbers), typeof(workspace)}(A.dims, numbers, workspace)
end
function Adapt.adapt_structure(to, tracker::ParticleBlockTracker)
    ParticleBlockTracker(adapt(to, tracker.blockids), adapt(to, tracker.counts))
end
function Adapt.adapt_structure(to, workspace::BlockSparsityWorkspace)
    BlockSparsityWorkspace(adapt(to, workspace.occupied), adapt(to, workspace.active), adapt(to, workspace.tracker))
end
function Adapt.adapt_structure(to, A::SpIndices{dim, L}) where {dim, L}
    numbers = adapt(to, blocknumbering(A))
    workspace = adapt(to, sparsity_workspace(A))
    SpIndices{dim, L, typeof(numbers), typeof(workspace)}(A.dims, numbers, workspace)
end
function KernelAbstractions.get_backend(A::SpIndices)
    get_backend(blocknumbering(A))
end

# SpArray
function Adapt.adapt_structure(to, A::SpArray)
    SpArray(adapt(to, get_data(A)), adapt(to, get_spinds(A)), A.shared_spinds)
end
function KernelAbstractions.get_backend(A::SpArray)
    backend = get_backend(A.data)
    @assert get_backend(A.spinds) == backend
    backend
end

# HybridArray
function Adapt.adapt_structure(to::AbstractDevice, A::HybridArray)
    parent′ = adapt(to, parent(A))
    HybridArray(parent′, flatten(parent′), get_device(parent′))
end
function Adapt.adapt_structure(to, A::HybridArray)
    HybridArray(adapt(to, parent(A)), adapt(to, flatten(A)), get_device(A))
end
