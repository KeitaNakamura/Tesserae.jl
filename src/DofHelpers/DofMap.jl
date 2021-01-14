"""
    DofMap(dims...)

Create function-like object for dof mapping.
`DofMap` also behave like a bool array to activate/deactivate indices.
To finalize activations and perform numbering dofs, use [`count!(::DofMap)`](@ref).

- [`count!(::DofMap)`](@ref)
- [`(DofMap)(index...; [dof])`](@ref)
- [`DofHelpers.map(::DofMap, inds; [dof])`](@ref)
- [`DofHelpers.filter(::DofMap, inds)`](@ref)
"""
struct DofMap{dim} <: AbstractArray{Bool, dim}
    indices::Array{Int, dim}
end

DofMap(dims::Tuple{Vararg{Int}}) = DofMap(fill(-1, dims))
DofMap(dims::Int...) = DofMap(dims)

Base.size(dofmap::DofMap) = size(dofmap.indices)
Base.IndexStyle(::Type{<: DofMap}) = IndexLinear()

Base.getindex(dofmap::DofMap, i::Int) = (@_propagate_inbounds_meta; dofmap.indices[i] !== -1)
Base.setindex!(dofmap::DofMap, v::Bool, i::Int) = (@_propagate_inbounds_meta; dofmap.indices[i] = ifelse(v, 0, -1))

Base.fill!(dofmap::DofMap, v::Bool) = (fill!(dofmap.indices, ifelse(v, 0, -1)); dofmap)

"""
    ndofs(::DofMap; [dof])

Return total number of dofs.
"""
function ndofs(dofmap::DofMap; dof::Int = 1)
    i = findlast(>(0), dofmap.indices)
    i === nothing && return 0
    dofmap.indices[i] * dof
end

"""
    count!(::DofMap)

Count active indices and numbering dofs.
Returned integer is the number of active indices.

# Examples
```jldoctest
julia> dofmap = DofMap(5, 5)
5×5 DofMap{2}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> dofmap[1:2, 2:3] .= true; dofmap
5×5 DofMap{2}:
 0  1  1  0  0
 0  1  1  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> count!(dofmap)
4
```
"""
function Base.count!(dofmap::DofMap)
    count = 0
    for i in eachindex(dofmap)
        @inbounds dofmap.indices[i] = (dofmap[i] ? count += 1 : -1)
    end
    count
end

"""
    (::DofMap)(index...; [dof])

Return dof from given nodal `index`.
Use [`DofHelpers.map(::DofMap, indices::AbstractArray)`](@ref) for multiple indices.

# Examples
```jldoctest
julia> dofmap = DofMap(5, 5)
5×5 DofMap{2}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> dofmap[1:2, 2:3] .= true; dofmap
5×5 DofMap{2}:
 0  1  1  0  0
 0  1  1  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> count!(dofmap)
4

julia> dofmap(1, 1) === nothing
true

julia> dofmap(2, 2)
2

julia> dofmap(2, 2, dof = 1)
2:2

julia> dofmap(2, 2, dof = 2)
3:4
```
"""
function (dofmap::DofMap)(I...; dof = nothing)
    @_propagate_inbounds_meta
    j = dofmap.indices[I...]
    j == -1 && return nothing
    dof === nothing && return j
    start = dof*(j-1)
    (start+1):(start+dof)
end

"""
    DofHelpers.map(::DofMap, indices::AbstractArray; [dof])
    DofHelpers.map!(::DofMap, dofs::Vector{Int}, indices::AbstractArray; [dof])

Map nodal `indices` to dof indices.
This is almost the same behavior as performing `dofmap(index)` at each elemnt of `indices` but `nothing` is skipped.

# Examples
```jldoctest
julia> dofmap = DofMap(5, 5)
5×5 DofMap{2}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> dofmap[1:2, 2:3] .= true; dofmap
5×5 DofMap{2}:
 0  1  1  0  0
 0  1  1  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> count!(dofmap)
4

julia> DofHelpers.map(dofmap, CartesianIndices((1:2, 1:2)); dof = 2)
4-element Array{Int64,1}:
 1
 2
 3
 4

julia> DofHelpers.map(dofmap, [CartesianIndex(1, 2), CartesianIndex(2, 3)]; dof = 2)
4-element Array{Int64,1}:
 1
 2
 7
 8
```
"""
function map(dofmap::DofMap, inds::AbstractArray; dof::Int = 1)
    out = Int[]
    map!(dofmap, out, inds; dof)
    out
end

function map!(dofmap::DofMap, dofs::Vector{Int}, inds::AbstractArray; dof::Int = 1)
    linear = view(dofmap.indices, inds) # checkbounds as well
    resize!(dofs, length(inds) * dof)
    count = 0
    allactive = true
    @inbounds for i in eachindex(linear)
        j = linear[i]
        if j == -1
            allactive = false
            continue
        end
        for d in 1:dof
            dofs[count += 1] = dof*(j-1) + d
        end
    end
    resize!(dofs, count)
    allactive
end

"""
    DofHelpers.filter(::DofMap, inds::AbstractArray)
    DofHelpers.filter!(::DofMap, output::Vector, inds::AbstractArray)

This is the same as `inds[dofmap[inds]]` but more effective.

# Examples
```jldoctest
julia> dofmap = DofMap(5, 5)
5×5 DofMap{2}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> dofmap[1:2, 2:3] .= true; dofmap
5×5 DofMap{2}:
 0  1  1  0  0
 0  1  1  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0

julia> count!(dofmap)
4

julia> DofHelpers.filter(dofmap, CartesianIndices((1:2, 1:2)))
2-element Array{CartesianIndex{2},1}:
 CartesianIndex(1, 2)
 CartesianIndex(2, 2)
```
"""
function filter(dofmap::DofMap, inds::AbstractArray)
    out = eltype(inds)[]
    filter!(dofmap, out, inds)
    out
end

function filter!(dofmap::DofMap, output::Vector, inds::AbstractArray)
    resize!(output, length(inds))
    count = 0
    allactive = true
    @inbounds for i in eachindex(inds)
        j = inds[i]
        if dofmap.indices[j] == -1
            allactive = false
            continue
        end
        output[count += 1] = j
    end
    resize!(output, count)
    allactive
end
