# Shared cell-weight storage and geometric updates

generate_basis_weights(::Type{T}, mesh::AbstractCellMesh, dims...; kwargs...) where {T} = _generate_basis_weights(T, basis(mesh), mesh, _todims(dims...); kwargs...)
generate_basis_weights(mesh::AbstractCellMesh, dims...; kwargs...) = generate_basis_weights(Float64, mesh, dims...; kwargs...)

function allocate_cell_basis_values(::Type{Prop}, basis, ::Val{dim}; derivative::Order) where {Prop <: NamedTuple, dim}
    check_cell_basis_derivative(derivative)
    _allocate_basis_values(MArray{Tuple{nsupportnodes(basis)}}, Prop, Val(dim); derivative)
end

# FEM/IGA: every quadrature point of a cell shares that cell's support.
mutable struct CellSupportMatrix{T, V <: AbstractVector{T}} <: AbstractMatrix{T}
    const dims::Dims{2}
    cellsupports::V
end

function CellSupportMatrix(supports::V, nq::Int, ncells::Int) where {T, V <: AbstractVector{T}}
    ncells == length(supports) || throw(DimensionMismatch("the second basis-weight dimension must equal the number of cells"))
    CellSupportMatrix{T, V}((nq, ncells), supports)
end

Base.size(A::CellSupportMatrix) = A.dims
Base.IndexStyle(::Type{<: CellSupportMatrix}) = IndexCartesian()
cellsupports(A::CellSupportMatrix) = getfield(A, :cellsupports)
@inline function Base.getindex(A::CellSupportMatrix, q::Int, cell::Int)
    @boundscheck checkbounds(A, q, cell)
    @inbounds cellsupports(A)[cell]
end
function set_cellsupports!(A::CellSupportMatrix{T, V}, supports::V) where {T, V <: AbstractVector{T}}
    length(supports) == size(A, 2) || throw(DimensionMismatch("the number of cells must match the second basis-weight dimension"))
    setfield!(A, :cellsupports, supports)
    A
end

function check_cell_basis_derivative(::Order{k}) where {k}
    0 ≤ k ≤ 1 || throw(ArgumentError("FEM and IGA basis weights support derivative=Order(0) or Order(1) only, got Order($k)"))
    nothing
end

function _check_quadrature_inputs(is_domain, weights, points, geometry, measure, normal)
    rule = quadrature_rule(points)
    _check_quadrature_rule(rule, geometry)
    size(weights) == size(points) || throw(DimensionMismatch("basis weights and quadrature points must have the same dimensions"))
    size(points) == (length(rule.points), ncells(geometry)) || throw(DimensionMismatch("quadrature points must have dimensions (rule points, geometry cells)"))
    isnothing(measure) || size(weights) == size(measure) || throw(DimensionMismatch("measure and basis weights must have the same dimensions"))
    is_domain && !isnothing(normal) && throw(ArgumentError("normal is only valid for boundary integration"))
    isnothing(normal) || size(weights) == size(normal) || throw(DimensionMismatch("normal and basis weights must have the same dimensions"))
    if is_domain && !isempty(weights)
        bw = first(weights)
        derivative_order(bw) isa Order{0} && throw(ArgumentError("domain basis weights must store first derivatives"))
        length(eltype(nodal_basis_values(bw, Order(1)))) == length(eltype(geometry)) || throw(DimensionMismatch("basis-gradient and geometry dimensions must match"))
    end
    rule
end

@inline function _set_quadrature_data!(::Val{:domain}, bw, measure, normal, q, cell, N, dNdξ, J, weight)
    set_values!(bw, (N, dNdξ .⊡ Ref(inv(J))))
    if measure !== nothing
        measure[q,cell] = weight * sqrt(det(J'J))
    end
end

@inline function _set_quadrature_data!(::Val{:boundary}, bw, measure, normal, q, cell, N, dNdξ, J, weight)
    metric = J'J
    if derivative_order(bw) isa Order{0}
        set_values!(bw, (N,))
    else
        # Contracting with the left pseudoinverse gives the ambient tangential gradient.
        set_values!(bw, (N, dNdξ .⊡ Ref(inv(metric) * J')))
    end
    if measure !== nothing
        measure[q,cell] = weight * sqrt(det(metric))
    end
    if normal !== nothing
        n = _get_normal(J)
        normal[q,cell] = n / norm(n)
    end
end

_get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
_get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
_get_normal(J::Mat{3,1}) = throw(ArgumentError("normal is not defined for a 3D line element"))
