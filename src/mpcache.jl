struct MPCache{dim, T, Tshape <: ShapeValues{dim, T}}
    shapevalues::Vector{Tshape}
    gridsize::NTuple{dim, Int}
    npoints::Base.RefValue{Int}
    pointsinblock::Array{Vector{Int}, dim}
    spat::Array{Bool, dim}
end

function MPCache(grid::Grid{dim, T}, xₚ::AbstractVector) where {dim, T}
    checkshapefunction(grid)
    npoints = length(xₚ)
    shapevalues = [ShapeValues{dim, T}(grid.shapefunction) for _ in 1:npoints]
    MPCache(shapevalues, size(grid), Ref(npoints), pointsinblock(grid, xₚ), fill(false, size(grid)))
end

gridsize(cache::MPCache) = cache.gridsize
npoints(cache::MPCache) = cache.npoints[]
pointsinblock(cache::MPCache) = cache.pointsinblock

function reorder_pointstate!(pointstate::AbstractVector, cache::MPCache)
    @assert length(pointstate) == npoints(cache)
    inds = Vector{Int}(undef, length(pointstate))
    cnt = 1
    for block in pointsinblock(cache)
        @inbounds for i in eachindex(block)
            inds[cnt] = block[i]
            block[i] = cnt
            cnt += 1
        end
    end
    @inbounds @. pointstate = pointstate[inds]
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

function pointsinblock!(ptsinblk::AbstractArray{Vector{Int}, dim}, grid::Grid{dim}, xₚ::AbstractVector) where {dim}
    empty!.(ptsinblk)
    @inbounds for p in eachindex(xₚ)
        I = whichblock(grid, xₚ[p])
        I === nothing && continue
        push!(ptsinblk[I], p)
    end
    ptsinblk
end

function pointsinblock(grid::Grid, xₚ::AbstractVector)
    ptsinblk = Array{Vector{Int}}(undef, blocksize(grid))
    @inbounds @simd for i in eachindex(ptsinblk)
        ptsinblk[i] = Int[]
    end
    pointsinblock!(ptsinblk, grid, xₚ)
end

function sparsity_pattern!(spat::Array{Bool}, grid::Grid, xₚ::AbstractVector, hₚ::AbstractVector, ptsinblk::AbstractArray{Vector{Int}})
    @assert size(spat) == size(grid)
    fill!(spat, false)
    for blocks in threadsafe_blocks(size(grid))
        Threads.@threads for blockindex in blocks
            for p in ptsinblk[blockindex]
                inds = neighboring_nodes(grid, xₚ[p], hₚ[p])
                @inbounds spat[inds] .= true
            end
        end
    end
    @inbounds Threads.@threads for I in eachindex(grid)
        if isinbound(grid, I)
            spat[I] = false
        end
    end
    spat
end

function sparsity_pattern!(spat::Array{Bool}, grid::Grid, pointstate, ptsinblk::AbstractArray{Vector{Int}})
    hₚ = Fill(active_length(grid.shapefunction), length(pointstate))
    sparsity_pattern!(spat, grid, pointstate.x, hₚ, ptsinblk)
    spat
end

function sparsity_pattern!(spat::Array{Bool}, grid::Grid{<: Any, <: Any, GIMP}, pointstate, ptsinblk::AbstractArray{Vector{Int}})
    hₚ = LazyDotArray(rₚ -> active_length(grid.shapefunction, rₚ ./ gridsteps(grid)), pointstate.r)
    sparsity_pattern!(spat, grid, pointstate.x, hₚ, ptsinblk)
    spat
end

function update_shapevalues!(shapevalues::Vector{<: ShapeValues}, grid::Grid, pointstate, spat::AbstractArray{Bool, dim}, p::Int) where {dim}
    update!(shapevalues[p], grid, pointstate.x[p], spat)
end

function update_shapevalues!(shapevalues::Vector{<: GIMPValues}, grid::Grid, pointstate, spat::AbstractArray{Bool, dim}, p::Int) where {dim}
    update!(shapevalues[p], grid, pointstate.x[p], pointstate.r[p], spat)
end

function update!(cache::MPCache{dim}, grid::Grid{dim}, pointstate) where {dim}
    @assert size(grid) == gridsize(cache)

    shapevalues = cache.shapevalues
    pointsinblock = cache.pointsinblock
    spat = cache.spat

    cache.npoints[] = length(pointstate)
    allocate!(i -> eltype(shapevalues)(), shapevalues, length(pointstate))

    pointsinblock!(pointsinblock, grid, pointstate.x)
    sparsity_pattern!(spat, grid, pointstate, pointsinblock)

    Threads.@threads for p in eachindex(pointstate)
        @inbounds update_shapevalues!(shapevalues, grid, pointstate, spat, p)
    end

    gridstate = grid.state
    gridstate.spat .= spat
    reinit!(gridstate)

    cache
end

##################
# point_to_grid! #
##################

@generated function _point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray, N}}, shapevalues::ShapeValues, p::Int) where {N}
    exps = [:(add!(gridstates[$i], I.i, res[$i])) for i in 1:N]
    quote
        @inbounds @simd for i in eachindex(shapevalues)
            it = shapevalues[i]
            I = it.index
            res = p2g(it, p, I)
            $(exps...)
        end
    end
end

function point_to_grid!(p2g, gridstates::Tuple{Vararg{AbstractArray}}, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    @assert all(==(gridsize(cache)), size.(gridstates))
    pointmask !== nothing && @assert length(pointmask) == npoints(cache)
    for blocks in threadsafe_blocks(gridsize(cache))
        Threads.@threads for blockindex in blocks
            @inbounds for p in pointsinblock(cache)[blockindex]
                pointmask !== nothing && !pointmask[p] && continue
                _point_to_grid!(p2g, gridstates, cache.shapevalues[p], p)
            end
        end
    end
    gridstates
end

function point_to_grid!(p2g, gridstate::AbstractArray, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    point_to_grid!((gridstate,), cache, pointmask) do it, p, I
        @_inline_meta
        @_propagate_inbounds_meta
        (p2g(it, p, I),)
    end
end

function point_to_grid!(p2g, gridstate::AbstractArray, grid::Grid, xₚ::AbstractVector)
    point_to_grid!((gridstate,), grid, xₚ) do it, p, I
        @_inline_meta
        @_propagate_inbounds_meta
        (p2g(it, p, I),)
    end
end

@inline function stress_to_force(coordinate_system::Symbol, N, ∇N, x::Vec{2}, σ::SymmetricSecondOrderTensor{3})
    f = Tensor2D(σ) ⋅ ∇N
    if coordinate_system == :axisymmetric
        @inbounds f += Vec(1,0)*σ[3,3]*N/x[1]
    end
    f
end
@inline function stress_to_force(::Symbol, N, ∇N, x::Vec{3}, σ::SymmetricSecondOrderTensor{3})
    σ ⋅ ∇N
end

for (ShapeFunctionType, ShapeValuesType) in ((BSpline, BSplineValues),
                                             (GIMP, GIMPValues))
    @eval function default_point_to_grid!(
            grid::Grid{<: Any, <: Any, <: $ShapeFunctionType},
            pointstate,
            cache::MPCache{<: Any, <: Any, <: $ShapeValuesType}
        )
        point_to_grid!((grid.state.m, grid.state.v, grid.state.f), cache) do it, p, i
            @_inline_meta
            @_propagate_inbounds_meta
            N = it.N
            ∇N = it.∇N
            xₚ = pointstate.x[p]
            mₚ = pointstate.m[p]
            Vₚ = pointstate.V[p]
            vₚ = pointstate.v[p]
            σₚ = pointstate.σ[p]
            bₚ = pointstate.b[p]
            xᵢ = grid[i]
            m = mₚ * N
            v = m * vₚ
            f = -Vₚ * stress_to_force(grid.coordinate_system, N, ∇N, xₚ, σₚ) + m * bₚ
            m, v, f
        end
        @dot_threads grid.state.v /= grid.state.m
        @. grid.state.v_n = grid.state.v
        grid
    end
end

function default_point_to_grid!(grid::Grid{<: Any, <: Any, <: WLS},
                                pointstate,
                                cache::MPCache{<: Any, <: Any, <: WLSValues})
    P = polynomial(grid.shapefunction)
    point_to_grid!((grid.state.m, grid.state.w, grid.state.v, grid.state.f), cache) do it, p, i
        @_inline_meta
        @_propagate_inbounds_meta
        N = it.N
        ∇N = it.∇N
        w = it.w
        xₚ = pointstate.x[p]
        mₚ = pointstate.m[p]
        Vₚ = pointstate.V[p]
        Cₚ = pointstate.C[p]
        σₚ = pointstate.σ[p]
        bₚ = pointstate.b[p]
        xᵢ = grid[i]
        m = mₚ * N
        v = w * Cₚ ⋅ P(xᵢ - xₚ)
        f = -Vₚ * stress_to_force(grid.coordinate_system, N, ∇N, xₚ, σₚ) + m * bₚ
        m, w, v, f
    end
    @dot_threads grid.state.v /= grid.state.w
    grid
end

##################
# grid_to_point! #
##################

@generated function _grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector, N}}, shapevalues::ShapeValues, p::Int) where {N}
    quote
        vals = tuple($([:(zero(eltype(pointstates[$i]))) for i in 1:N]...))
        @inbounds @simd for i in eachindex(shapevalues)
            it = shapevalues[i]
            I = it.index
            res = g2p(it, I, p)
            vals = tuple($([:(vals[$i] + res[$i]) for i in 1:N]...))
        end
        $([:(setindex!(pointstates[$i], vals[$i], p)) for i in 1:N]...)
    end
end

function grid_to_point!(g2p, pointstates::Tuple{Vararg{AbstractVector}}, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    @assert all(==(npoints(cache)), length.(pointstates))
    pointmask !== nothing && @assert length(pointmask) == npoints(cache)
    @inbounds Threads.@threads for p in 1:npoints(cache)
        pointmask !== nothing && !pointmask[p] && continue
        _grid_to_point!(g2p, pointstates, cache.shapevalues[p], p)
    end
    pointstates
end

function grid_to_point!(g2p, pointstate::AbstractVector, cache::MPCache, pointmask::Union{AbstractVector{Bool}, Nothing} = nothing)
    grid_to_point!((pointstate,), cache, pointmask) do it, I, p
        @_inline_meta
        @_propagate_inbounds_meta
        (g2p(it, I, p),)
    end
end

function grid_to_point!(g2p, pointstate::AbstractVector, grid::Grid, xₚ::AbstractVector)
    grid_to_point!((pointstate,), grid, xₚ) do it, I, p
        @_inline_meta
        @_propagate_inbounds_meta
        (g2p(it, I, p),)
    end
end

@inline function velocity_gradient(coordinate_system::Symbol, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    ∇v = Poingr.Tensor3D(∇v)
    if coordinate_system == :axisymmetric
        @inbounds ∇v += @Mat([0 0 0; 0 0 0; 0 0 v[1]/x[1]])
    end
    ∇v
end
@inline function velocity_gradient(::Symbol, x::Vec{3}, v::Vec{3}, ∇v::SecondOrderTensor{3})
    ∇v
end

for (ShapeFunctionType, ShapeValuesType) in ((BSpline, BSplineValues),
                                             (GIMP, GIMPValues))
    @eval function default_grid_to_point!(
            pointstate,
            grid::Grid{dim, <: Any, <: $ShapeFunctionType},
            cache::MPCache{dim, <: Any, <: $ShapeValuesType},
            dt::Real
        ) where {dim}
        @inbounds Threads.@threads for p in 1:npoints(cache)
            shapevalues = cache.shapevalues[p]
            T = eltype(pointstate.v[p])
            dvₚ = zero(Vec{dim, T})
            vₚ = zero(Vec{dim, T})
            ∇vₚ = zero(Mat{dim, dim, T})
            @simd for i in eachindex(shapevalues)
                it = shapevalues[i]
                I = it.index
                N = it.N
                ∇N = it.∇N
                dvᵢ = grid.state.v[I] - grid.state.v_n[I]
                vᵢ = grid.state.v[I]
                dvₚ += N * dvᵢ
                vₚ += N * vᵢ
                ∇vₚ += vᵢ ⊗ ∇N
            end
            pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, ∇vₚ)
            pointstate.v[p] += dvₚ
            pointstate.x[p] += vₚ * dt
        end
        pointstate
    end
end

function default_grid_to_point!(pointstate,
                                grid::Grid{dim, <: Any, <: WLS},
                                cache::MPCache{dim, <: Any, <: WLSValues},
                                dt::Real) where {dim}
    P = polynomial(grid.shapefunction)
    p0 = P(zero(Vec{dim, Int}))
    ∇p0 = P'(zero(Vec{dim, Int}))
    grid_to_point!(pointstate.C, cache) do it, i, p
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
        pointstate.v[p] = vₚ
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, xₚ, vₚ, Cₚ ⋅ ∇p0)
        pointstate.x[p] = xₚ + vₚ * dt
    end
    pointstate
end

function smooth_trace_of_velocity_gradient!(pointstate::StructVector, grid::Grid, cache::MPCache)
    point_to_grid!(grid.state.tr_∇v, cache) do it, p, i
        @_inline_meta
        @_propagate_inbounds_meta
        ∇vₚ = pointstate.∇v[p]
        it.N * pointstate.m[p] * tr(∇vₚ)
    end
    @dot_threads grid.state.tr_∇v /= grid.state.m
    grid_to_point!(pointstate.tr_∇v, cache) do it, i, p
        @_inline_meta
        @_propagate_inbounds_meta
        it.N * grid.state.tr_∇v[i]
    end
end
