struct MPSpace{dim, T, Tshape <: ShapeValues{dim, T}}
    shapevalues::Vector{Tshape}
    gridsize::NTuple{dim, Int}
    gridindices::Vector{Vector{Index{dim}}}
    pointsinblock::Array{Vector{Int}, dim}
    nearsurface::BitVector
end

function MPSpace(grid::Grid{dim, T}, xₚ::AbstractVector) where {dim, T}
    checkshapefunction(grid)
    npoints = length(xₚ)
    shapevalues = [ShapeValues(T, grid.shapefunction) for _ in 1:npoints]
    gridindices = [Index{dim}[] for _ in 1:npoints]
    MPSpace(shapevalues, size(grid), gridindices, pointsinblock(grid, xₚ), falses(npoints))
end

npoints(space::MPSpace) = length(space.nearsurface)
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
    checkshapefunction(grid)
    @assert size(grid) == gridsize(space)

    allocate!(i -> eltype(space.shapevalues)(), space.shapevalues, length(xₚ))
    allocate!(i -> Index{dim}[], space.gridindices, length(xₚ))
    resize!(space.nearsurface, length(xₚ))

    gridstate = grid.state
    pointsinblock!(space.pointsinblock, grid, xₚ)

    spat = sparsity_pattern(grid, xₚ, space.pointsinblock; exclude)
    space.nearsurface .= false
    @inbounds Threads.@threads for p in eachindex(xₚ)
        x = xₚ[p]
        gridindices = space.gridindices[p]
        space.nearsurface[p] = neighboring_nodes!(gridindices, grid, x, spat)
        reinit!(space.shapevalues[p], grid, x, gridindices)
    end

    gridstate.spat .= spat
    reinit!(gridstate)

    space
end

##################
# point_to_grid! #
##################

@generated function _point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray, N}}, shapevalues::ShapeValues, gridindices::Vector{<: Index}, p::Int) where {N}
    exps = [:(gridstates[$i][I] += res[$i]) for i in 1:N]
    quote
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
            @inbounds for p in space.pointsinblock[blockindex]
                pointmask !== nothing && !pointmask[p] && continue
                _point_to_grid!(p2g, gridstates, space.shapevalues[p], space.gridindices[p], p)
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

function default_point_to_grid!(grid::Grid{<: Any, <: Any, <: WLS},
                                pointstate::StructVector,
                                space::MPSpace{<: Any, <: Any, <: WLSValues},
                                coord_system::Symbol)
    P = polynomial(grid.shapefunction)
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

@generated function _grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector, N}}, shapevalues::ShapeValues, gridindices::Vector{<: Index}, p::Int) where {N}
    quote
        vals = tuple($([:(zero(eltype(pointstates[$i]))) for i in 1:N]...))
        @inbounds @simd for i in eachindex(shapevalues, gridindices)
            it = shapevalues[i]
            I = gridindices[i]
            res = g2p(it, I, p)
            vals = tuple($([:(vals[$i] + res[$i]) for i in 1:N]...))
        end
        $([:(setindex!(pointstates[$i], vals[$i], p)) for i in 1:N]...)
    end
end

function grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector}}, space::MPSpace, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    @assert all(==(npoints(space)), length.(pointstates))
    pointmask !== nothing && @assert length(pointmask) == npoints(space)
    Threads.@threads for p in 1:npoints(space)
        pointmask !== nothing && !pointmask[p] && continue
        _grid_to_point!(g2p, pointstates, space.shapevalues[p], space.gridindices[p], p)
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
                                grid::Grid{dim, <: Any, <: WLS},
                                space::MPSpace{dim, <: Any, <: WLSValues},
                                dt::Real,
                                coord_system::Symbol) where {dim}
    P = polynomial(grid.shapefunction)
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
