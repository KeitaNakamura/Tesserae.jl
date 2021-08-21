struct MPSpace{dim, T, Tf <: ShapeFunction{dim}, Tshape <: ShapeValues{dim, T}}
    F::Tf
    shapevalues::Vector{Tshape}
    gridsize::NTuple{dim, Int}
    gridindices::Vector{Vector{GridIndex{dim}}}
    pointsinblock::Array{Vector{Int}, dim}
    nearsurface::BitVector
end

function MPSpace(F::ShapeFunction, grid::Grid{dim, T}, xₚ::AbstractVector) where {dim, T}
    npoints = length(xₚ)
    shapevalues = [ShapeValues(T, F) for _ in 1:npoints]
    gridindices = [GridIndex{dim}[] for _ in 1:npoints]
    MPSpace(F, shapevalues, size(grid), gridindices, pointsinblock(grid, xₚ), falses(npoints))
end

npoints(space::MPSpace) = length(space.shapevalues)
gridsize(space::MPSpace) = space.gridsize

function reordering_pointstate!(pointstate::AbstractVector, space::MPSpace)
    inds = Vector{Int}(undef, length(pointstate))
    cnt = 1
    for block in space.pointsinblock
        @inbounds for i in eachindex(block)
            inds[cnt] = block[i]
            block[i] = cnt
            cnt += 1
        end
    end
    @. pointstate = pointstate[inds]
    pointstate
end

function allocate!(f, x::Vector, n::Integer)
    len = length(x)
    if n > len # growend
        resize!(x, n)
        @simd for i in len+1:n
            @inbounds x[i] = f(i)
        end
    end
    x
end

function reinit!(space::MPSpace{dim}, grid::Grid{dim}, xₚ::AbstractVector; exclude = nothing) where {dim}
    @assert size(grid) == gridsize(space)

    allocate!(i -> eltype(space.shapevalues)(), space.shapevalues, length(xₚ))
    allocate!(i -> GridIndex{dim}[], space.gridindices, length(xₚ))
    resize!(space.nearsurface, length(xₚ))

    gridstate = grid.state
    pointsinblock!(space.pointsinblock, grid, xₚ)

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
    @inbounds Threads.@threads for p in eachindex(xₚ)
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

##################
# point_to_grid! #
##################

@generated function _point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray, N}}, space::MPSpace, p::Int) where {N}
    exps = [:(gridstates[$i][I] += res[$i]) for i in 1:N]
    quote
        shapevalues = space.shapevalues[p]
        gridindices = space.gridindices[p]
        @inbounds @simd for i in eachindex(shapevalues, gridindices)
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

function stress_to_force(coord_system::Symbol, N, ∇N, x::Vec, σ::SymmetricSecondOrderTensor{3})
    f = Tensor2D(σ) ⋅ ∇N
    if coord_system == :axisymmetric
        return f + Vec(1,0) * σ[3,3] * N / x[1]
    end
    f
end

function default_point_to_grid!(grid::Grid,
                                pointstate::StructVector,
                                space::MPSpace{<: Any, <: Any, <: WLS},
                                coord_system::Symbol)
    P = polynomial(space.F)
    point_to_grid!((grid.state.m, grid.state.w, grid.state.v, grid.state.f), space) do it, p, i
        @_inline_meta
        @_propagate_inbounds_meta
        N = it.N
        ∇N = it.∇N
        w = it.w
        xₚ = pointstate.x[p]
        mₚ = pointstate.m[p]
        V0ₚ = pointstate.V0[p]
        Fₚ = pointstate.F[p]
        Cₚ = pointstate.C[p]
        σₚ = pointstate.σ[p]
        bₚ = pointstate.b[p]
        xᵢ = grid[i]
        m = mₚ * N
        v = w * Cₚ ⋅ P(xᵢ - xₚ)
        f = -(V0ₚ*det(Fₚ)) * stress_to_force(coord_system, N, ∇N, xₚ, σₚ) + m * bₚ
        m, w, v, f
    end
    @. grid.state.v /= grid.state.w
    grid
end

##################
# grid_to_point! #
##################

function _grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector, N}}, space::MPSpace, p::Int) where {N}
    vals = zero.(eltype.(pointstates))
    shapevalues = space.shapevalues[p]
    gridindices = space.gridindices[p]
    @inbounds @simd for i in eachindex(shapevalues, gridindices)
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

function velocity_gradient(coord_system::Symbol, x::Vec, v::Vec, ∇v::SecondOrderTensor{2})
    L = Poingr.Tensor3D(∇v)
    if coord_system == :axisymmetric
        return L + @Mat [0.0 0.0 0.0
                         0.0 0.0 0.0
                         0.0 0.0 v[1] / x[1]]
    end
    L
end

function default_grid_to_point!(pointstate::StructVector,
                                grid::Grid,
                                space::MPSpace{dim, <: Any, <: WLS},
                                dt::Real,
                                coord_system::Symbol) where {dim}
    P = polynomial(space.F)
    p0 = P(zero(Vec{dim, Int}))
    ∇p0 = P'(zero(Vec{dim, Int}))
    grid_to_point!(pointstate.C, space) do it, i, p
        @_inline_meta
        @_propagate_inbounds_meta
        w = it.w
        M⁻¹ = it.M⁻¹
        grid.state.v[i] ⊗ (w * M⁻¹ ⋅ P(grid[i] - pointstate.x[p]))
    end
    @inbounds Threads.@threads for p in eachindex(pointstate)
        Cₚ = pointstate.C[p]
        xₚ = pointstate.x[p]
        vₚ = Cₚ ⋅ p0
        ∇vₚ = velocity_gradient(coord_system, xₚ, vₚ, Cₚ ⋅ ∇p0)
        Fₚ = pointstate.F[p]
        pointstate.v[p] = vₚ
        pointstate.∇v[p] = ∇vₚ
        pointstate.F[p] = Fₚ + dt*(∇vₚ ⋅ Fₚ)
        pointstate.x[p] = xₚ + vₚ * dt
    end
    pointstate
end
