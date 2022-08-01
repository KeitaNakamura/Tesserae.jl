struct MPSpace{T, dim, F <: Interpolation, C <: CoordinateSystem, V <: MPValues{dim, T}}
    interp::F
    grid::Grid{T, dim, C}
    spat::Array{Bool, dim}
    mpvals::Vector{V}
    ptspblk::Array{Vector{Int}, dim}
    npts::RefValue{Int}
    stamp::RefValue{Float64}
end

# constructors
function MPSpace(interp::Interpolation, grid::Grid{T, dim}, xₚ::AbstractVector{<: Vec{dim}}) where {dim, T}
    spat = fill(false, size(grid))
    npts = length(xₚ)
    mpvals = [MPValues{dim, T}(interp) for _ in 1:npts]
    MPSpace(interp, grid, spat, mpvals, pointsperblock(grid, xₚ), Ref(npts), Ref(NaN))
end
MPSpace(interp::Interpolation, grid::Grid, pointstate::AbstractVector) = MPSpace(interp, grid, pointstate.x)

# helper functions
gridsize(space::MPSpace) = size(space.grid)
num_points(space::MPSpace) = space.npts[]
get_interpolation(space::MPSpace) = space.interp
get_grid(space::MPSpace) = space.grid
get_sppattern(space::MPSpace) = space.spat
get_mpvalues(space::MPSpace) = space.mpvals
get_mpvalues(space::MPSpace, i::Int) = (@_propagate_inbounds_meta; space.mpvals[i])
get_pointsperblock(space::MPSpace) = space.ptspblk
get_stamp(space::MPSpace) = space.stamp[]

# reorder_pointstate!
function reorder_pointstate!(pointstate::AbstractVector, ptspblk::Array)
    @assert length(pointstate) == sum(length, ptspblk)
    inds = Vector{Int}(undef, length(pointstate))
    cnt = 1
    for blocks in threadsafe_blocks(size(ptspblk))
        @inbounds for blockindex in blocks
            block = ptspblk[blockindex]
            for i in eachindex(block)
                inds[cnt] = block[i]
                block[i] = cnt
                cnt += 1
            end
        end
    end
    @inbounds @. pointstate = pointstate[inds]
    pointstate
end
reorder_pointstate!(pointstate::AbstractVector, space::MPSpace) = reorder_pointstate!(pointstate, get_pointsperblock(space))

# pointsperblock!
function pointsperblock!(ptspblk::AbstractArray{Vector{Int}}, grid::Grid, xₚ::AbstractVector)
    empty!.(ptspblk)
    @inbounds for p in 1:length(xₚ)
        I = whichblock(grid, xₚ[p])
        I === nothing || push!(ptspblk[I], p)
    end
    ptspblk
end
function pointsperblock(grid::Grid, xₚ::AbstractVector)
    ptspblk = Array{Vector{Int}}(undef, blocksize(size(grid)))
    @inbounds @simd for i in eachindex(ptspblk)
        ptspblk[i] = Int[]
    end
    pointsperblock!(ptspblk, grid, xₚ)
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

function update!(space::MPSpace, pointstate::AbstractVector; exclude::Union{Nothing, AbstractArray{Bool}} = nothing)
    space.npts[]  = length(pointstate)
    space.stamp[] = time()

    mpvals = get_mpvalues(space)
    allocate!(i -> eltype(mpvals)(), mpvals, length(pointstate))

    update_sppattern!(space, pointstate; exclude)
    Threads.@threads for p in 1:length(pointstate)
        @inbounds update!(mpvals[p], get_grid(space), LazyRow(pointstate, p), get_sppattern(space))
    end

    space
end

function update_sppattern!(space::MPSpace, pointstate::AbstractVector; exclude::Union{Nothing, AbstractArray{Bool}} = nothing)
    # update `pointsperblock`
    pointsperblock!(get_pointsperblock(space), get_grid(space), pointstate.x)

    # update sparsity pattern
    spat = get_sppattern(space)
    fill!(spat, false)
    eachpoint_blockwise_parallel(space) do p
        @_inline_propagate_inbounds_meta
        inds = gridindices(get_interpolation(space), get_grid(space), LazyRow(pointstate, p))
        spat[inds] .= true
    end

    # handle excluded domain
    if exclude !== nothing
        @. spat &= !exclude
        eachpoint_blockwise_parallel(space) do p
            @_inline_propagate_inbounds_meta
            inds = gridindices(get_grid(space), pointstate.x[p], 1)
            spat[inds] .= true
        end
    end
    spat
end

function update_sppattern!(gridstate::SpArray, space::MPSpace)
    @assert is_parent(gridstate)
    @assert size(gridstate) == gridsize(space)
    update_sppattern!(gridstate, get_sppattern(space))
    set_stamp!(gridstate, get_stamp(space))
    gridstate
end

function eachpoint_blockwise_parallel(f, space::MPSpace)
    for blocks in threadsafe_blocks(blocksize(gridsize(space)))
        Threads.@threads for blockindex in blocks
            @inbounds for p in get_pointsperblock(space)[blockindex]
                f(p)
            end
        end
    end
end

##################
# point_to_grid! #
##################

function check_gridstate(gridstate::AbstractArray, space::MPSpace)
    @assert size(gridstate) == gridsize(space)
end
function check_gridstate(gridstate::SpArray, space::MPSpace)
    @assert size(gridstate) == gridsize(space)
    @assert get_stamp(gridstate) == get_stamp(space)
end

function point_to_grid!(p2g, gridstates, mps::MPValues)
    @_inline_propagate_inbounds_meta
    @simd for i in 1:length(mps)
        I = gridindices(mps, i)
        map_tuple(add!, gridstates, p2g(mps[i], I), I)
    end
end

function point_to_grid!(p2g, gridstates, space::MPSpace; zeroinit::Bool = true)
    map_tuple(check_gridstate, gridstates, space)
    zeroinit && map_tuple(fillzero!, gridstates)
    eachpoint_blockwise_parallel(space) do p
        @_inline_propagate_inbounds_meta
        point_to_grid!(
            (mp, I) -> (@_inline_propagate_inbounds_meta; p2g(mp, p, I)),
            gridstates,
            get_mpvalues(space, p),
        )
    end
    gridstates
end

function point_to_grid!(p2g, gridstates, space::MPSpace, pointmask::AbstractVector{Bool}; zeroinit::Bool = true)
    map_tuple(check_gridstate, gridstates, space)
    @assert length(pointmask) == num_points(space)
    zeroinit && map_tuple(fillzero!, gridstates)
    eachpoint_blockwise_parallel(space) do p
        @_inline_propagate_inbounds_meta
        pointmask[p] && point_to_grid!(
            (mp, I) -> (@_inline_propagate_inbounds_meta; p2g(mp, p, I)),
            gridstates,
            get_mpvalues(space, p),
        )
    end
    gridstates
end

##################
# grid_to_point! #
##################

function check_pointstate(pointstate::AbstractVector, space::MPSpace)
    @assert length(pointstate) == num_points(space)
end

function grid_to_point(g2p, mps::MPValues)
    @_inline_propagate_inbounds_meta
    vals = g2p(first(mps), gridindices(mps, 1))
    @simd for i in 2:length(mps)
        I = gridindices(mps, i)
        vals = map_tuple(+, vals, g2p(mps[i], I))
    end
    vals
end

function grid_to_point(g2p, space::MPSpace)
    LazyDotArray(1:num_points(space)) do p
        @_inline_propagate_inbounds_meta
        grid_to_point(
            (mp, I) -> (@_inline_propagate_inbounds_meta; g2p(mp, I, p)),
            get_mpvalues(space, p)
        )
    end
end

function grid_to_point!(g2p, pointstates, space::MPSpace)
    map_tuple(check_pointstate, pointstates, space)
    results = grid_to_point(g2p, space)
    Threads.@threads for p in 1:num_points(space)
        @inbounds map_tuple(setindex!, pointstates, results[p], p)
    end
end

function grid_to_point!(g2p, pointstates, space::MPSpace, pointmask::AbstractVector{Bool})
    map_tuple(check_pointstate, pointstates, space)
    @assert length(pointmask) == num_points(space)
    results = grid_to_point(g2p, space)
    Threads.@threads for p in 1:num_points(space)
        @inbounds pointmask[p] && map_tuple(setindex!, pointstates, results[p], p)
    end
end

######################
# smooth_pointstate! #
######################

@generated function safe_inv(x::Mat{dim, dim, T, L}) where {dim, T, L}
    exps = fill(:z, L-1)
    quote
        @_inline_meta
        z = zero(T)
        isapproxzero(det(x)) ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x)
        # Tensorial.rank(x) != dim ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x) # this is very slow but stable
    end
end

function smooth_pointstate!(vals::AbstractVector, Vₚ::AbstractVector, gridstate::AbstractArray, space::MPSpace)
    @assert length(vals) == length(Vₚ) == num_points(space)
    grid = get_grid(space)
    basis = PolynomialBasis{1}()
    point_to_grid!((gridstate.poly_coef, gridstate.poly_mat), space) do mp, p, i
        @_inline_propagate_inbounds_meta
        P = value(basis, mp.xp - grid[i])
        VP = (mp.N * Vₚ[p]) * P
        VP * vals[p], VP ⊗ P
    end
    @dot_threads gridstate.poly_coef = safe_inv(gridstate.poly_mat) ⋅ gridstate.poly_coef
    grid_to_point!(vals, space) do mp, i, p
        @_inline_propagate_inbounds_meta
        P = value(basis, mp.xp - grid[i])
        mp.N * (P ⋅ gridstate.poly_coef[i])
    end
end
