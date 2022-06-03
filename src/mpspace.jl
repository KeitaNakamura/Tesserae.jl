struct MPSpace{T, dim, F <: Interpolation, CS, MV <: MPValues{dim, T}}
    interp::F
    grid::Grid{T, dim, CS}
    spat::Array{Bool, dim}
    mpvalues::Vector{MV}
    ptsinblk::Array{Vector{Int}, dim}
    npts::Base.RefValue{Int}
    stamp::Base.RefValue{Float64}
end

# constructors
function MPSpace(interp::Interpolation, grid::Grid{T, dim}, xₚ::AbstractVector{<: Vec{dim}}) where {dim, T}
    spat = fill(false, size(grid))
    npts = length(xₚ)
    mpvalues = [MPValues{dim, T}(interp) for _ in 1:npts]
    MPSpace(interp, grid, spat, mpvalues, pointsinblock(grid, xₚ), Ref(npts), Ref(NaN))
end
MPSpace(interp::Interpolation, grid::Grid, pointstate::AbstractVector) = MPSpace(interp, grid, pointstate.x)

# helper functions
gridsize(space::MPSpace) = size(space.grid)
num_points(space::MPSpace) = space.npts[]
get_pointsinblock(space::MPSpace) = space.ptsinblk
get_sparsitypattern(space::MPSpace) = space.spat
get_interpolation(space::MPSpace) = space.interp
get_stamp(space::MPSpace) = space.stamp[]

# reorder_pointstate!
function reorder_pointstate!(pointstate::AbstractVector, ptsinblk::Array)
    @assert length(pointstate) == sum(length, ptsinblk)
    inds = Vector{Int}(undef, length(pointstate))
    cnt = 1
    for blocks in threadsafe_blocks(@. $size(ptsinblk) << BLOCK_UNIT + 1)
        @inbounds for blockindex in blocks
            block = ptsinblk[blockindex]
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
reorder_pointstate!(pointstate::AbstractVector, space::MPSpace) = reorder_pointstate!(pointstate, get_pointsinblock(space))

# pointsinblock!
function pointsinblock!(ptsinblk::AbstractArray{Vector{Int}}, grid::Grid, xₚ::AbstractVector)
    empty!.(ptsinblk)
    @inbounds for p in 1:length(xₚ)
        I = whichblock(grid, xₚ[p])
        I === nothing || push!(ptsinblk[I], p)
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

function update_sparsitypattern!(spat::Array{Bool}, interp::Interpolation, grid::Grid, pointstate::AbstractVector, ptsinblk::AbstractArray{Vector{Int}}; exclude)
    @assert size(spat) == size(grid)
    fill!(spat, false)
    for blocks in threadsafe_blocks(size(grid))
        Threads.@threads for blockindex in blocks
            for p in ptsinblk[blockindex]
                inds = neighbornodes(interp, grid, LazyRow(pointstate, p))
                @inbounds spat[inds] .= true
            end
        end
    end
    if exclude !== nothing
        @. spat &= !exclude
        for blocks in threadsafe_blocks(size(grid))
            Threads.@threads for blockindex in blocks
                for p in ptsinblk[blockindex]
                    inds = neighbornodes(grid, pointstate.x[p], 1)
                    @inbounds spat[inds] .= true
                end
            end
        end
    end
    spat
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

function update!(space::MPSpace, pointstate; exclude::Union{Nothing, AbstractArray{Bool}} = nothing)
    grid = space.grid
    mpvalues = space.mpvalues
    ptsinblk = space.ptsinblk
    spat = space.spat

    space.npts[]  = length(pointstate)
    space.stamp[] = time()
    allocate!(i -> eltype(mpvalues)(), mpvalues, length(pointstate))

    pointsinblock!(ptsinblk, grid, pointstate.x)
    update_sparsitypattern!(spat, get_interpolation(space), grid, pointstate, ptsinblk; exclude)

    Threads.@threads for p in 1:length(pointstate)
        @inbounds update!(mpvalues[p], grid, LazyRow(pointstate, p), spat)
    end

    space
end

function update_sparsitypattern!(spat::SpArray, MPSpace::MPSpace)
    update_sparsitypattern!(spat, get_sparsitypattern(MPSpace))
    spat.stamp[] = get_stamp(MPSpace)
    spat
end

function eachpoint_blockwise_parallel(f, space::MPSpace)
    for blocks in threadsafe_blocks(gridsize(space))
        Threads.@threads for blockindex in blocks
            @inbounds for p in get_pointsinblock(space)[blockindex]
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
            space.mpvalues[p],
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
            space.mpvalues[p],
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
            space.mpvalues[p]
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
    grid = space.grid
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
