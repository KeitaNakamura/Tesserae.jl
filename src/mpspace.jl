struct MPSpace{dim, T, Tf <: ShapeFunction{dim}, Tshape <: ShapeValues{dim, T}}
    F::Tf
    shapevalues::Vector{Tshape}
    gridsize::NTuple{dim, Int}
    gridindices::Vector{Vector{GridIndex{dim}}}
    pointsinblock::Array{Vector{Int}, dim}
    nearsurface::BitVector
end

function MPSpace(F::ShapeFunction, grid::Grid{dim, T}, npoints::Int) where {dim, T}
    shapevalues = [ShapeValues(T, F) for _ in 1:npoints]
    gridindices = [GridIndex{dim}[] for _ in 1:npoints]
    MPSpace(F, shapevalues, size(grid), gridindices, pointsinblock(grid, 1:0), falses(npoints))
end

npoints(space::MPSpace) = length(space.shapevalues)
gridsize(space::MPSpace) = space.gridsize

# reinitialize pointsinblock and reordering pointstate
function reinit!(pointsinblock::Array, pointstate::StructVector, grid::Grid)
    pointsinblock!(pointsinblock, grid, pointstate.x)
    # inds = Vector{Int}(undef, length(pointstate))
    # cnt = 1
    # for block in pointsinblock
        # @inbounds for i in eachindex(block)
            # inds[cnt] = block[i]
            # block[i] = cnt
            # cnt += 1
        # end
    # end
    # @. pointstate = pointstate[inds]
    nothing
end

function reinit!(space::MPSpace, grid::Grid, pointstate::AbstractVector; exclude = nothing)
    @assert size(grid) == gridsize(space)
    @assert length(pointstate) == npoints(space)

    gridstate = grid.state
    reinit!(space.pointsinblock, pointstate, grid)
    xₚ = pointstate.x

    point_radius = support_length(space.F)
    mask = gridstate.mask
    mask .= false
    for color in coloringblocks(gridsize(space))
        Threads.@threads for blockindex in color
            @inbounds for p in space.pointsinblock[blockindex]
                inds = neighboring_nodes(grid, xₚ[p], point_radius)
                mask[inds] .= true
            end
        end
    end

    if exclude !== nothing
        @inbounds Threads.@threads for I in eachindex(grid)
            x = grid[I]
            exclude(x) && (mask[I] = false)
        end
        for color in coloringblocks(gridsize(space))
            Threads.@threads for blockindex in color
                @inbounds for p in space.pointsinblock[blockindex]
                    inds = neighboring_nodes(grid, xₚ[p], 1)
                    mask[inds] .= true
                end
            end
        end
    end

    space.nearsurface .= false
    @inbounds Threads.@threads for p in eachindex(xₚ, space.shapevalues, space.gridindices)
        x = xₚ[p]
        gridindices = space.gridindices[p]
        inds = neighboring_nodes(grid, x, point_radius)
        cnt = 0
        for I in inds
            mask[I] && (cnt += 1)
        end
        resize!(gridindices, cnt)
        cnt = 0
        for I in inds
            if mask[I]
                gridindices[cnt+=1] = GridIndex(grid, I)
            else
                space.nearsurface[p] = true
            end
        end
        reinit!(space.shapevalues[p], grid, x, gridindices)
    end

    reinit!(grid.state)

    space
end

@generated function _point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray, N}}, space::MPSpace, p::Int) where {N}
    exps = [:(gridstates[$i][I] += res[$i]) for i in 1:N]
    quote
        shapevalues = space.shapevalues[p]
        gridindices = space.gridindices[p]
        @inbounds for i in eachindex(shapevalues, gridindices)
            it = shapevalues[i]
            I = gridindices[i]
            res = p2g(it, p, I)
            $(exps...)
        end
    end
end
function point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray}}, space::MPSpace, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    @assert all(==(gridsize(space)), size.(gridstates))
    pointmask !== nothing && @assert length(pointmask) == npoints(space)
    for color in coloringblocks(gridsize(space))
        Threads.@threads for blockindex in color
            for p in space.pointsinblock[blockindex]
                pointmask !== nothing && !pointmask[p] && continue
                _point_to_grid!(p2g, gridstates, space, p)
            end
        end
    end
    gridstates
end
function point_to_grid!(p2g, gridstate::AbstractArray, space::MPSpace, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    point_to_grid!((gridstate,), space, pointmask) do it, p, I
        @_inline_meta
        @_propagate_inbounds_meta
        (p2g(it, p, I),)
    end
end

function _grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector, N}}, space::MPSpace, p::Int) where {N}
    vals = zero.(eltype.(pointstates))
    shapevalues = space.shapevalues[p]
    gridindices = space.gridindices[p]
    @inbounds for i in eachindex(shapevalues, gridindices)
        it = shapevalues[i]
        I = gridindices[i]
        res = g2p(it, I, p)
        vals = vals .+ res
    end
    setindex!.(pointstates, vals, p)
end
function grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector}}, space::MPSpace, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    @assert all(==(npoints(space)), length.(pointstates))
    pointmask !== nothing && @assert length(pointmask) == npoints(space)
    Threads.@threads for p in 1:npoints(space)
        pointmask !== nothing && !pointmask[p] && continue
        _grid_to_point!(g2p, pointstates, space, p)
    end
    pointstates
end
function grid_to_point!(g2p, pointstate::AbstractVector, space::MPSpace, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    grid_to_point!((pointstate,), space, pointmask) do it, I, p
        @_inline_meta
        @_propagate_inbounds_meta
        (g2p(it, I, p),)
    end
end
