# P2G
struct P2G_Normal    end
struct P2G_Taylor    end
struct P2G_WLS       end
struct P2G_AffinePIC end
struct P2G_Default   end
# G2P
struct G2P_FLIP      end
struct G2P_PIC       end
struct G2P_WLS       end
struct G2P_AffinePIC end
struct G2P_Default   end

struct Transfer{P2G, G2P}
    P2G::P2G
    G2P::G2P
    Transfer{P2G, G2P}() where {P2G, G2P} = new(P2G(), G2P())
end
# default
@pure DefaultTransfer() = Transfer{P2G_Default, G2P_Default}()

# supported transfer combinations
const FLIP = Transfer{P2G_Normal, G2P_FLIP}
const PIC  = Transfer{P2G_Normal, G2P_PIC}
const TFLIP = Transfer{P2G_Taylor, G2P_FLIP}
const TPIC  = Transfer{P2G_Taylor, G2P_PIC}
const APIC  = Transfer{P2G_AffinePIC, G2P_AffinePIC}

########################
# default combinations #
########################

# use FLIP by default
@pure P2G_default(::Interpolation) = P2G_Normal()
@pure G2P_default(::Interpolation) = G2P_FLIP()

# WLS
@pure P2G_default(::WLS) = P2G_WLS()
@pure G2P_default(::WLS) = G2P_WLS()

################
# P2G transfer #
################

point_to_grid!(t::Transfer, args...) = point_to_grid!(t.P2G, args...)

function point_to_grid!(::P2G_Default, gridstate::AbstractArray, pointstate::AbstractVector, space::MPSpace, dt::Real)
    P2G = P2G_default(get_interpolation(space))
    point_to_grid!(P2G, gridstate, pointstate, space, dt)
end

function point_to_grid!(::P2G_Normal, gridstate::AbstractArray, pointstate::AbstractVector, space::MPSpace, dt::Real)
    grid = get_grid(space)
    point_to_grid!((gridstate.m, gridstate.v_n, gridstate.v), space) do mp, p, i
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        xₚ = pointstate.x[p]
        mₚ = pointstate.m[p]
        Vₚ = pointstate.V[p]
        vₚ = pointstate.v[p]
        σₚ = pointstate.σ[p]
        bₚ = pointstate.b[p]
        xᵢ = grid[i]
        m = mₚ * N
        mv = m * vₚ
        f = -Vₚ * stress_to_force(grid.coordinate_system, N, ∇N, xₚ, σₚ) + m * bₚ
        m, mv, mv + dt*f
    end
    @dot_threads gridstate.v_n /= gridstate.m
    @dot_threads gridstate.v /= gridstate.m
    gridstate
end

function point_to_grid!(::P2G_AffinePIC, gridstate::AbstractArray, pointstate::AbstractVector, space::MPSpace, dt::Real)
    grid = get_grid(space)
    D = grid_to_point(space) do mp, i, p
        @_inline_propagate_inbounds_meta
        N = mp.N
        xᵢ = grid[i]
        xₚ = pointstate.x[p]
        N * (xᵢ - xₚ) ⊗ (xᵢ - xₚ)
    end
    point_to_grid!((gridstate.m, gridstate.v_n, gridstate.v), space) do mp, p, i
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        xₚ = pointstate.x[p]
        mₚ = pointstate.m[p]
        Vₚ = pointstate.V[p]
        vₚ = pointstate.v[p]
        Bₚ = pointstate.B[p]
        σₚ = pointstate.σ[p]
        bₚ = pointstate.b[p]
        Cₚ = Bₚ ⋅ inv(D[p])
        xᵢ  = grid[i]
        m = mₚ * N
        mv = m * (vₚ + Cₚ ⋅ (xᵢ - xₚ))
        f = -Vₚ * stress_to_force(grid.coordinate_system, N, ∇N, xₚ, σₚ) + m * bₚ
        m, mv, mv + dt*f
    end
    @dot_threads gridstate.v_n /= gridstate.m
    @dot_threads gridstate.v /= gridstate.m
    gridstate
end

function point_to_grid!(::P2G_Taylor, gridstate::AbstractArray, pointstate::AbstractVector, space::MPSpace{<: Any, dim}, dt::Real) where {dim}
    grid = get_grid(space)
    point_to_grid!((gridstate.m, gridstate.v_n, gridstate.v), space) do mp, p, i
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        xₚ  = pointstate.x[p]
        mₚ  = pointstate.m[p]
        Vₚ  = pointstate.V[p]
        vₚ  = pointstate.v[p]
        ∇vₚ = pointstate.∇v[p]
        σₚ  = pointstate.σ[p]
        bₚ  = pointstate.b[p]
        xᵢ  = grid[i]
        m = mₚ * N
        mv = m * (vₚ + @Tensor(∇vₚ[1:dim, 1:dim]) ⋅ (xᵢ - xₚ))
        f = -Vₚ * stress_to_force(grid.coordinate_system, N, ∇N, xₚ, σₚ) + m * bₚ
        m, mv, mv + dt*f
    end
    @dot_threads gridstate.v_n /= gridstate.m
    @dot_threads gridstate.v /= gridstate.m
    gridstate
end

function point_to_grid!(::P2G_WLS, gridstate::AbstractArray, pointstate::AbstractVector, space::MPSpace, dt::Real)
    grid = get_grid(space)
    P = get_basis(get_interpolation(space))
    point_to_grid!((gridstate.m, gridstate.v_n, gridstate.v), space) do mp, p, i
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        xₚ = pointstate.x[p]
        mₚ = pointstate.m[p]
        Vₚ = pointstate.V[p]
        Cₚ = pointstate.C[p]
        σₚ = pointstate.σ[p]
        bₚ = pointstate.b[p]
        xᵢ = grid[i]
        m = mₚ * N
        mv = m * Cₚ ⋅ value(P, xᵢ - xₚ)
        f = -Vₚ * stress_to_force(grid.coordinate_system, N, ∇N, xₚ, σₚ) + m * bₚ
        m, mv, mv + dt*f
    end
    @dot_threads gridstate.v_n /= gridstate.m
    @dot_threads gridstate.v /= gridstate.m
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

function grid_to_point!(::G2P_Default, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace, dt::Real)
    G2P = G2P_default(get_interpolation(space))
    grid_to_point!(G2P, pointstate, gridstate, space, dt)
end

function grid_to_point!(::G2P_FLIP, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace, dt::Real)
    grid = get_grid(space)
    pointvalues = grid_to_point(space) do mp, i, p
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        dvᵢ = gridstate.v[i] - gridstate.v_n[i]
        vᵢ = gridstate.v[i]
        N*dvᵢ, N*vᵢ, vᵢ⊗∇N
    end
    @inbounds Threads.@threads for p in 1:num_points(space)
        dvₚ, vₚ, ∇vₚ = pointvalues[p]
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, ∇vₚ)
        pointstate.v[p] += dvₚ
        pointstate.x[p] += vₚ * dt
    end
    pointstate
end

function grid_to_point!(::G2P_PIC, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace, dt::Real)
    grid = get_grid(space)
    pointvalues = grid_to_point(space) do mp, i, p
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        vᵢ = gridstate.v[i]
        vᵢ*N, vᵢ⊗∇N
    end
    @inbounds Threads.@threads for p in 1:num_points(space)
        vₚ, ∇vₚ = pointvalues[p]
        xₚ = pointstate.x[p]
        pointstate.v[p] = vₚ
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, xₚ, vₚ, ∇vₚ)
        pointstate.x[p] = xₚ + vₚ * dt
    end
    pointstate
end

function grid_to_point!(::G2P_AffinePIC, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace, dt::Real)
    grid = get_grid(space)
    pointvalues = grid_to_point(space) do mp, i, p
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        vᵢ = gridstate.v[i]
        xᵢ = grid[i]
        xₚ = pointstate.x[p]
        v = vᵢ * N
        ∇v = vᵢ ⊗ ∇N
        v, ∇v, v ⊗ (xᵢ - xₚ)
    end
    @inbounds Threads.@threads for p in 1:num_points(space)
        vₚ, ∇vₚ, Bₚ = pointvalues[p]
        xₚ = pointstate.x[p]
        pointstate.v[p] = vₚ
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, xₚ, vₚ, ∇vₚ)
        pointstate.x[p] = xₚ + vₚ * dt
        pointstate.B[p] = Bₚ
    end
    pointstate
end

function grid_to_point!(::G2P_WLS, pointstate::AbstractVector, gridstate::AbstractArray, space::MPSpace{<: Any, dim}, dt::Real) where {dim}
    grid = get_grid(space)
    P = get_basis(get_interpolation(space))
    p0 = value(P, zero(Vec{dim, Int}))
    ∇p0 = gradient(P, zero(Vec{dim, Int}))
    grid_to_point!(pointstate.C, space) do mp, i, p
        @_inline_propagate_inbounds_meta
        w = mp.w
        Minv = mp.Minv
        gridstate.v[i] ⊗ (w * Minv ⋅ value(P, grid[i] - pointstate.x[p]))
    end
    @inbounds Threads.@threads for p in 1:length(pointstate)
        Cₚ = pointstate.C[p]
        xₚ = pointstate.x[p]
        vₚ = Cₚ ⋅ p0
        pointstate.v[p] = vₚ
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, xₚ, vₚ, Cₚ ⋅ ∇p0)
        pointstate.x[p] = xₚ + vₚ * dt
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
