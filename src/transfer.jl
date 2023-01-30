# P2G
abstract type P2G_Transfer end
struct P2G_Normal     <: P2G_Transfer end
struct P2G_Taylor     <: P2G_Transfer end
struct P2G_WLS        <: P2G_Transfer end
struct P2G_AffineFLIP <: P2G_Transfer end
struct P2G_AffinePIC  <: P2G_Transfer end
# G2P
abstract type G2P_Transfer end
struct G2P_FLIP       <: G2P_Transfer end
struct G2P_PIC        <: G2P_Transfer end
struct G2P_WLS        <: G2P_Transfer end
struct G2P_AffineFLIP <: G2P_Transfer end
struct G2P_AffinePIC  <: G2P_Transfer end

struct Transfer{P2G <: P2G_Transfer, G2P <: G2P_Transfer}
    P2G::P2G
    G2P::G2P
end
Transfer{P2G, G2P}() where {P2G, G2P} = Transfer(P2G(), G2P())

# supported transfer combinations
const FLIP  = Transfer{P2G_Normal, G2P_FLIP}
const PIC   = Transfer{P2G_Normal, G2P_PIC}
const TFLIP = Transfer{P2G_Taylor, G2P_FLIP}
const TPIC  = Transfer{P2G_Taylor, G2P_PIC}
const AFLIP = Transfer{P2G_AffineFLIP, G2P_AffineFLIP}
const APIC  = Transfer{P2G_AffinePIC, G2P_AffinePIC}

const WLSTransfer = Transfer{P2G_WLS, G2P_WLS}

########################
# default combinations #
########################

@pure Transfer(interp::Interpolation) = Transfer(P2G_default(interp), G2P_default(interp))

# use FLIP by default
@pure P2G_default(::Interpolation) = P2G_Normal()
@pure G2P_default(::Interpolation) = G2P_FLIP()

# WLS
@pure P2G_default(::WLS) = P2G_WLS()
@pure G2P_default(::WLS) = G2P_WLS()

################
# P2G transfer #
################

function check_states(gridstate::AbstractArray, pointstate::AbstractVector, space::MPSpace)
    @assert length(pointstate) == num_points(space)
    @assert size(gridstate) == gridsize(space)
    check_gridstate(gridstate, space)
end

function check_gridstate(gridstate::StructArray, space::MPSpace)
end
function check_gridstate(gridstate::SpArray, space::MPSpace)
    if get_stamp(gridstate) != get_stamp(space)
        # check to use @inbounds for `SpArray`
        error("`update_sparsity_pattern!(gridstate::SpArray, space::MPSpace)` must be executed before `point_to_grid!` and `grid_to_point!`")
    end
end

point_to_grid!(t::Transfer, args...) = point_to_grid!(t.P2G, args...)

function point_to_grid!(::P2G_Normal, gridstate::GridStateArray, pointstate::PointStateVector, space::MPSpace, dt::Real)
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)
    fillzero!(gridstate.m)
    fillzero!(gridstate.vⁿ)
    fillzero!(gridstate.v)

    eachpoint_blockwise_parallel(space) do p
        @inbounds begin
            xₚ = pointstate.x[p]
            mₚ = pointstate.m[p]
            Vₚ = pointstate.V[p]
            vₚ = pointstate.v[p]
            σₚ = pointstate.σ[p]
            bₚ = pointstate.b[p]
            mp = get_mpvalue(space, p)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                f = -Vₚ*stress_to_force(grid.coordinate_system, N, mp.∇N[j], xₚ, σₚ) + N*mₚ*bₚ
                gridstate.m[i]  += N*mₚ
                gridstate.vⁿ[i] += N*mₚ*vₚ
                gridstate.v[i]  += dt*f
            end
        end
    end

    @dot_threads gridstate.v = ((gridstate.vⁿ + gridstate.v) / gridstate.m) * !iszero(gridstate.m)
    @dot_threads gridstate.vⁿ = (gridstate.vⁿ / gridstate.m) * !iszero(gridstate.m)

    gridstate
end

function point_to_grid!(::Union{P2G_AffinePIC, P2G_AffineFLIP}, gridstate::GridStateArray, pointstate::PointStateVector, space::MPSpace{dim, T}, dt::Real) where {dim, T}
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)
    fillzero!(gridstate.m)
    fillzero!(gridstate.vⁿ)
    fillzero!(gridstate.v)

    eachpoint_blockwise_parallel(space) do p
        @inbounds begin
            mp = get_mpvalue(space, p)
            Dₚ = zero(Mat{dim, dim, T})
            xₚ = pointstate.x[p]
            for (j, i) in enumerate(get_nodeindices(space, p))
                xᵢ = grid[i]
                N = mp.N[j]
                Dₚ += N * (xᵢ - xₚ) ⊗ (xᵢ - xₚ)
            end
            mₚ = pointstate.m[p]
            Vₚ = pointstate.V[p]
            vₚ = pointstate.v[p]
            Bₚ = pointstate.B[p]
            σₚ = pointstate.σ[p]
            bₚ = pointstate.b[p]
            Cₚ = Bₚ ⋅ inv(Dₚ)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                xᵢ  = grid[i]
                f = -Vₚ*stress_to_force(grid.coordinate_system, N, mp.∇N[j], xₚ, σₚ) + N*mₚ*bₚ
                gridstate.m[i]  += N*mₚ
                gridstate.vⁿ[i] += N*mₚ*(vₚ + Cₚ⋅(xᵢ - xₚ))
                gridstate.v[i]  += dt*f
            end
        end
    end

    @dot_threads gridstate.v = ((gridstate.vⁿ + gridstate.v) / gridstate.m) * !iszero(gridstate.m)
    @dot_threads gridstate.vⁿ = (gridstate.vⁿ / gridstate.m) * !iszero(gridstate.m)

    gridstate
end

function point_to_grid!(::P2G_Taylor, gridstate::GridStateArray, pointstate::PointStateVector, space::MPSpace{dim}, dt::Real) where {dim}
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)
    fillzero!(gridstate.m)
    fillzero!(gridstate.vⁿ)
    fillzero!(gridstate.v)

    eachpoint_blockwise_parallel(space) do p
        @inbounds begin
            xₚ  = pointstate.x[p]
            mₚ  = pointstate.m[p]
            Vₚ  = pointstate.V[p]
            vₚ  = pointstate.v[p]
            ∇vₚ = pointstate.∇v[p]
            σₚ  = pointstate.σ[p]
            bₚ  = pointstate.b[p]
            mp = get_mpvalue(space, p)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                xᵢ = grid[i]
                f = -Vₚ*stress_to_force(grid.coordinate_system, N, mp.∇N[j], xₚ, σₚ) + N*mₚ*bₚ
                gridstate.m[i]  += N*mₚ
                gridstate.vⁿ[i] += N*mₚ*(vₚ + @Tensor(∇vₚ[1:dim, 1:dim]) ⋅ (xᵢ - xₚ))
                gridstate.v[i]  += dt*f
            end
        end
    end

    @dot_threads gridstate.v = ((gridstate.vⁿ + gridstate.v) / gridstate.m) * !iszero(gridstate.m)
    @dot_threads gridstate.vⁿ = (gridstate.vⁿ / gridstate.m) * !iszero(gridstate.m)

    gridstate
end

function point_to_grid!(::P2G_WLS, gridstate::GridStateArray, pointstate::PointStateVector, space::MPSpace, dt::Real)
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)
    fillzero!(gridstate.m)
    fillzero!(gridstate.vⁿ)
    fillzero!(gridstate.v)

    eachpoint_blockwise_parallel(space) do p
        @inbounds begin
            xₚ = pointstate.x[p]
            mₚ = pointstate.m[p]
            Vₚ = pointstate.V[p]
            Cₚ = pointstate.C[p]
            σₚ = pointstate.σ[p]
            bₚ = pointstate.b[p]
            mp = get_mpvalue(space, p)
            P = get_basis(mp)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                xᵢ = grid[i]
                f = -Vₚ*stress_to_force(grid.coordinate_system, N, mp.∇N[j], xₚ, σₚ) + N*mₚ*bₚ
                gridstate.m[i] += N*mₚ
                gridstate.vⁿ[i] += N*mₚ*Cₚ⋅value(P, xᵢ-xₚ)
                gridstate.v[i] += dt*f
            end
        end
    end

    @dot_threads gridstate.v = ((gridstate.vⁿ + gridstate.v) / gridstate.m) * !iszero(gridstate.m)
    @dot_threads gridstate.vⁿ = (gridstate.vⁿ / gridstate.m) * !iszero(gridstate.m)

    gridstate
end

@inline function stress_to_force(::PlaneStrain, N, ∇N, x::Vec{2}, σ::SymmetricSecondOrderTensor{3})
    Tensorial.resizedim(σ, Val(2)) ⋅ ∇N
end
@inline function stress_to_force(::Axisymmetric, N, ∇N, x::Vec{2}, σ::SymmetricSecondOrderTensor{3})
    @inbounds Tensorial.resizedim(σ, Val(2)) ⋅ ∇N + Vec(1,0) * (σ[3,3] * N / x[1])
end
@inline function stress_to_force(::ThreeDimensional, N, ∇N, x::Vec{3}, σ::SymmetricSecondOrderTensor{3})
    σ ⋅ ∇N
end

################
# G2P transfer #
################

grid_to_point!(t::Transfer, args...) = grid_to_point!(t.G2P, args...)

function grid_to_point!(::G2P_FLIP, pointstate::PointStateVector, gridstate::GridStateArray, space::MPSpace{dim}, dt::Real) where {dim}
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)

    @inbounds Threads.@threads for p in 1:num_points(space)
        dvₚ = zero(eltype(pointstate.v))
        vₚ  = zero(eltype(pointstate.v))
        ∇vₚ = @Tensor zero(eltype(pointstate.∇v))[1:dim, 1:dim]
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            N = mp.N[j]
            ∇N = mp.∇N[j]
            dvᵢ = gridstate.v[i] - gridstate.vⁿ[i]
            vᵢ = gridstate.v[i]
            dvₚ += N * dvᵢ
            vₚ  += N * vᵢ
            ∇vₚ += vᵢ ⊗ ∇N
        end
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, ∇vₚ)
        pointstate.v[p] += dvₚ
        pointstate.x[p] += vₚ * dt
    end

    pointstate
end

function grid_to_point!(::G2P_PIC, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace{dim}, dt::Real) where {dim}
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)

    @inbounds Threads.@threads for p in 1:num_points(space)
        vₚ  = zero(eltype(pointstate.v))
        ∇vₚ = @Tensor zero(eltype(pointstate.∇v))[1:dim, 1:dim]
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            N = mp.N[j]
            ∇N = mp.∇N[j]
            vᵢ = gridstate.v[i]
            vₚ  += vᵢ * N
            ∇vₚ += vᵢ ⊗ ∇N
        end
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, ∇vₚ)
        pointstate.v[p] = vₚ
        pointstate.x[p] += vₚ * dt
    end

    pointstate
end

function grid_to_point!(::G2P_AffineFLIP, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace{dim}, dt::Real) where {dim}
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)

    @inbounds Threads.@threads for p in 1:num_points(space)
        dvₚ = zero(eltype(pointstate.v))
        vₚ  = zero(eltype(pointstate.v))
        ∇vₚ = @Tensor zero(eltype(pointstate.∇v))[1:dim, 1:dim]
        Bₚ  = zero(eltype(pointstate.B))
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            N = mp.N[j]
            ∇N = mp.∇N[j]
            dvᵢ = gridstate.v[i] - gridstate.vⁿ[i]
            vᵢ = gridstate.v[i]
            xᵢ = grid[i]
            xₚ = pointstate.x[p]
            dvₚ += N * dvᵢ
            vₚ  += N * vᵢ
            ∇vₚ += vᵢ ⊗ ∇N
            Bₚ  += N * vᵢ ⊗ (xᵢ - xₚ)
        end
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, ∇vₚ)
        pointstate.v[p] += dvₚ
        pointstate.x[p] += vₚ * dt
        pointstate.B[p] = Bₚ
    end

    pointstate
end

function grid_to_point!(::G2P_AffinePIC, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace{dim}, dt::Real) where {dim}
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)

    @inbounds Threads.@threads for p in 1:num_points(space)
        vₚ  = zero(eltype(pointstate.v))
        ∇vₚ = @Tensor zero(eltype(pointstate.∇v))[1:dim, 1:dim]
        Bₚ = zero(eltype(pointstate.B))
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            N = mp.N[j]
            ∇N = mp.∇N[j]
            vᵢ = gridstate.v[i]
            xᵢ = grid[i]
            xₚ = pointstate.x[p]
            vₚ  += N * vᵢ
            ∇vₚ += vᵢ ⊗ ∇N
            Bₚ  += N * vᵢ ⊗ (xᵢ - xₚ)
        end
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, ∇vₚ)
        pointstate.v[p] = vₚ
        pointstate.x[p] += vₚ * dt
        pointstate.B[p] = Bₚ
    end

    pointstate
end

function grid_to_point!(::G2P_WLS, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace{dim}, dt::Real) where {dim}
    check_states(gridstate, pointstate, space)

    grid = get_grid(space)

    @inbounds Threads.@threads for p in 1:num_points(space)
        Cₚ = zero(eltype(pointstate.C))
        mp = get_mpvalue(space, p)
        P = get_basis(mp)
        p0 = value(P, zero(Vec{dim, Int}))
        ∇p0 = gradient(P, zero(Vec{dim, Int}))
        for (j, i) in enumerate(get_nodeindices(space, p))
            w = mp.w[j]
            Minv = mp.Minv
            Cₚ += gridstate.v[i] ⊗ (w * Minv ⋅ value(P, grid[i] - pointstate.x[p]))
        end
        vₚ = Cₚ ⋅ p0
        pointstate.C[p] = Cₚ
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, Cₚ ⋅ ∇p0)
        pointstate.v[p] = vₚ
        pointstate.x[p] += vₚ * dt
    end

    pointstate
end

@inline function velocity_gradient(::PlaneStrain, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    Tensorial.resizedim(∇v, Val(3)) # expaned entries are filled with zero
end
@inline function velocity_gradient(::Axisymmetric, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    @inbounds Tensorial.resizedim(∇v, Val(3)) + @Mat([0 0 0; 0 0 0; 0 0 v[1]/x[1]])
end
@inline function velocity_gradient(::ThreeDimensional, x::Vec{3}, v::Vec{3}, ∇v::SecondOrderTensor{3})
    ∇v
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

function smooth_pointstate!(vals::AbstractVector, xₚ::AbstractVector, Vₚ::AbstractVector, gridstate::AbstractArray, space::MPSpace)
    check_states(gridstate, vals, space)
    @assert length(vals) == length(xₚ) == length(Vₚ)

    grid = get_grid(space)
    basis = PolynomialBasis{1}()
    fillzero!(gridstate.poly_coef)
    fillzero!(gridstate.poly_mat)

    eachpoint_blockwise_parallel(space) do p
        @inbounds begin
            mp = get_mpvalue(space, p)
            for (j, i) in enumerate(get_nodeindices(space, p))
                N = mp.N[j]
                P = value(basis, xₚ[p] - grid[i])
                VP = (mp.N[j] * Vₚ[p]) * P
                gridstate.poly_coef[i] += VP * vals[p]
                gridstate.poly_mat[i]  += VP ⊗ P
            end
        end
    end

    @dot_threads gridstate.poly_coef = safe_inv(gridstate.poly_mat) ⋅ gridstate.poly_coef

    @inbounds Threads.@threads for p in 1:num_points(space)
        val = zero(eltype(vals))
        mp = get_mpvalue(space, p)
        for (j, i) in enumerate(get_nodeindices(space, p))
            P = value(basis, xₚ[p] - grid[i])
            val += mp.N[j] * (P ⋅ gridstate.poly_coef[i])
        end
        vals[p] = val
    end

    vals
end
