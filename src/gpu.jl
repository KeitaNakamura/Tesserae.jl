# -----------------------------------------------------------------------------
#  CPU/GPU transfers
# -----------------------------------------------------------------------------

# ---- entry points ----

# NOTE: `gpu` always tries to convert Float64 to Float32 (is this really good?)

function Adapt.adapt_storage(::CPUDevice, A::AbstractArray)
    get_device(A) isa CPUDevice ? A : Array(A)
end

# A method specialized on `to::AbstractDevice` is an explicit Tesserae transfer,
# while an unspecialized `adapt_structure(to, ...)` may serve other Adapt callers.
cpu(A) = A |> CPUDevice()
gpu(A) = A |> gpu_device(CastFloat32)
gpu_preserve(A) = A |> gpu_device(PreserveEltype)

# ---- special conversions ----

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

# ---- GPU compatibility ----

KernelAbstractions.get_backend(::BitArray) = CPU() # should be implemented in KernelAbstractions.jl

function Adapt.adapt_structure(to, mesh::CartesianMesh)
    axes = map(a -> adapt(to, a), mesh.axes)
    T = eltype(eltype(axes))
    CartesianMesh(axes, T(spacing(mesh)), T(spacing_inv(mesh)); block_size_log2=Val(block_size_log2(mesh)))
end
function KernelAbstractions.get_backend(mesh::CartesianMesh)
    @assert allequal(map(get_backend, mesh.axes))
    get_backend(mesh.axes[1])
end

function KernelAbstractions.get_backend(mesh::FEMesh)
    backend = get_backend(mesh.nodes)
    @assert get_backend(cellsupports(mesh)) == backend
    @assert get_backend(supportnodes(mesh)) == backend
    backend
end

KernelAbstractions.get_backend(mesh::IGAMesh) = get_backend(mesh.controlpoints)

Adapt.adapt_structure(to, points::QuadraturePoints) = QuadraturePoints(adapt(to, parent(points)), quadrature_rule(points))
KernelAbstractions.get_backend(points::QuadraturePoints) = get_backend(parent(points))

Adapt.adapt_structure(to, A::CellSupportMatrix) = CellSupportMatrix(adapt(to, cellsupports(A)), size(A)...)
KernelAbstractions.get_backend(A::CellSupportMatrix) = get_backend(cellsupports(A))

# The copy gets its own cleared flag: the choice is not carried across the move
# because the values are not either. Any other `to` -- notably a kernel launch --
# drops the flag, `select_weights` having already read it.
function Adapt.adapt_structure(to::AbstractDevice, weights::BasisWeightArray)
    b = basis(weights)
    vals = map(a -> adapt(to, a), getfield(weights, :vals))
    indices = adapt(to, getfield(weights, :indices))
    BasisWeightArray(b, vals, indices, derivative_order(weights), deferring_flag(b))
end
function Adapt.adapt_structure(to, weights::BasisWeightArray)
    b = basis(weights)
    vals = map(a -> adapt(to, a), getfield(weights, :vals))
    indices = adapt(to, getfield(weights, :indices))
    # A `DeferralState` is host-side bookkeeping and does not survive; a filter
    # in that slot is real data a deferred kernel reads, and travels with it.
    BasisWeightArray(b, vals, indices, derivative_order(weights), adapt(to, deferring_state(weights)))
end
function KernelAbstractions.get_backend(weights::BasisWeightArray)
    # Deferred value arrays report `nothing`, so the stored arrays and the support
    # indices decide, and any that do report must agree.
    backends = filter(!isnothing, map(get_backend, values(getfield(weights, :vals))))
    backend = get_backend(getfield(weights, :indices))
    @assert all(==(backend), backends)
    backend
end

function Adapt.adapt_structure(to::AbstractDevice, A::SpIndices{dim, L}) where {dim, L}
    numbers = adapt(to, blocknumbering(A))
    workspace = BlockSparsityWorkspace(numbers)
    SpIndices{dim, L, typeof(numbers), typeof(workspace)}(A.dims, numbers, workspace)
end
# A partition transfers by rebuilding its strategy for the target device; the
# CPU-side assignment state is not carried over, so `update!` must run after
# the transfer, exactly as after construction.
function Adapt.adapt_structure(to::GPUDevice, partition::ThreadPartition{<: BlockStrategy})
    ThreadPartition(GPUBlockStrategy(adapt(to, strategy(partition).mesh)))
end
Adapt.adapt_structure(::GPUDevice, partition::ThreadPartition{<: GPUBlockStrategy}) = partition
Adapt.adapt_structure(::GPUDevice, ::ThreadPartition{<: CellStrategy}) = error("ThreadPartition: FEM/IGA cell partitions are CPU-only")
function Adapt.adapt_structure(to::CPUDevice, partition::ThreadPartition{<: GPUBlockStrategy})
    ThreadPartition(BlockStrategy(adapt(to, strategy(partition).mesh)))
end

function Adapt.adapt_structure(to, tracker::ParticleBlockTracker)
    ParticleBlockTracker(adapt(to, tracker.blockids), adapt(to, tracker.counts))
end
function Adapt.adapt_structure(to, workspace::BlockSparsityWorkspace)
    BlockSparsityWorkspace(adapt(to, workspace.occupied), adapt(to, workspace.active), adapt(to, workspace.tracker),
                           adapt(to, workspace.active_count), adapt(to, workspace.changed))
end
function Adapt.adapt_structure(to, A::SpIndices{dim, L}) where {dim, L}
    numbers = adapt(to, blocknumbering(A))
    workspace = adapt(to, sparsity_workspace(A))
    SpIndices{dim, L, typeof(numbers), typeof(workspace)}(A.dims, numbers, workspace)
end
function KernelAbstractions.get_backend(A::SpIndices)
    get_backend(blocknumbering(A))
end

function Adapt.adapt_structure(to, A::SpArray)
    SpArray(adapt(to, get_data(A)), adapt(to, get_spinds(A)), A.shared_spinds)
end
function KernelAbstractions.get_backend(A::SpArray)
    backend = get_backend(A.data)
    @assert get_backend(A.spinds) == backend
    backend
end

function Adapt.adapt_structure(to::AbstractDevice, A::HybridArray)
    parent′ = adapt(to, parent(A))
    HybridArray(parent′, flatten(parent′), get_device(parent′))
end
function Adapt.adapt_structure(to, A::HybridArray)
    HybridArray(adapt(to, parent(A)), adapt(to, flatten(A)), get_device(A))
end

Adapt.adapt_structure(to, ::DeferralState) = nothing
