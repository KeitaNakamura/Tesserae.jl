struct P2G_Normal  end
struct P2G_Affine  end
struct P2G_Taylor  end
struct P2G_Default end
struct G2P_FLIP    end
struct G2P_PIC     end
struct G2P_Default end

struct Transfer{P2G, G2P}
    point_to_grid!::P2G
    grid_to_point!::G2P
end
Transfer() = Transfer(P2G_Default(), G2P_Default())
Transfer{P2G, G2P}() where {P2G, G2P} = Transfer(P2G(), G2P())

for (P2G, G2P) in Iterators.product((:Normal, :Affine, :Taylor), (:FLIP, :PIC))
    T = Symbol(:Transfer, P2G, G2P)
    P2G = Symbol(:P2G_, P2G)
    G2P = Symbol(:G2P_, G2P)
    @eval begin
        const $T = Transfer{$P2G, $G2P}
        export $T
    end
end

default_p2g(::BSpline) = P2G_Normal()
default_p2g(::GIMP) = P2G_Normal()
default_p2g(::KernelCorrection) = P2G_Taylor()

default_g2p(::BSpline) = G2P_FLIP()
default_g2p(::GIMP) = G2P_FLIP()
default_g2p(::KernelCorrection) = G2P_PIC()

################
# P2G transfer #
################

function (::P2G_Normal)(grid::Grid, pointstate, cache::MPCache, dt::Real)
    point_to_grid!((grid.state.m, grid.state.v_n, grid.state.v), cache) do mp, p, i
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
    @dot_threads grid.state.v_n /= grid.state.m
    @dot_threads grid.state.v /= grid.state.m
    grid
end

function (::P2G_Taylor)(grid::Grid{dim}, pointstate, cache::MPCache{dim}, dt::Real) where {dim}
    point_to_grid!((grid.state.m, grid.state.v_n, grid.state.v), cache) do mp, p, i
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
    @dot_threads grid.state.v_n /= grid.state.m
    @dot_threads grid.state.v /= grid.state.m
    grid
end

function (::P2G_Default)(grid::Grid{<: Any, <: Any, <: WLS}, pointstate, cache::MPCache{<: Any, <: Any, <: WLSValues}, dt::Real)
    P = basis_function(grid.interpolation)
    point_to_grid!((grid.state.m, grid.state.v_n, grid.state.v), cache) do mp, p, i
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
    @dot_threads grid.state.v_n /= grid.state.m
    @dot_threads grid.state.v /= grid.state.m
    grid
end

function (::P2G_Default)(grid::Grid, pointstate, cache::MPCache, dt::Real)
    P2G! = default_p2g(grid.interpolation)
    P2G!(grid, pointstate, cache, dt)
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

@inline function velocity_gradient(::PlaneStrain, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    Tensorial.resizedim(∇v, Val(3)) # expaned entries are filled with zero
end
@inline function velocity_gradient(::Axisymmetric, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    @inbounds Tensorial.resizedim(∇v, Val(3)) + @Mat([0 0 0; 0 0 0; 0 0 v[1]/x[1]])
end
@inline function velocity_gradient(::ThreeDimensional, x::Vec{3}, v::Vec{3}, ∇v::SecondOrderTensor{3})
    ∇v
end

function (::G2P_FLIP)(pointstate, grid::Grid, cache::MPCache, dt::Real)
    pointvalues = grid_to_point(cache) do mp, i, p
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        dvᵢ = grid.state.v[i] - grid.state.v_n[i]
        vᵢ = grid.state.v[i]
        N*dvᵢ, N*vᵢ, vᵢ⊗∇N
    end
    @inbounds Threads.@threads for p in 1:npoints(cache)
        dvₚ, vₚ, ∇vₚ = pointvalues[p]
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, pointstate.x[p], vₚ, ∇vₚ)
        pointstate.v[p] += dvₚ
        pointstate.x[p] += vₚ * dt
    end
    pointstate
end

function (::G2P_PIC)(pointstate, grid::Grid, cache::MPCache, dt::Real)
    pointvalues = grid_to_point(cache) do mp, i, p
        @_inline_propagate_inbounds_meta
        N = mp.N
        ∇N = mp.∇N
        vᵢ = grid.state.v[i]
        vᵢ*N, vᵢ⊗∇N
    end
    @inbounds Threads.@threads for p in 1:npoints(cache)
        vₚ, ∇vₚ = pointvalues[p]
        xₚ = pointstate.x[p]
        pointstate.v[p] = vₚ
        pointstate.∇v[p] = velocity_gradient(grid.coordinate_system, xₚ, vₚ, ∇vₚ)
        pointstate.x[p] = xₚ + vₚ * dt
    end
    pointstate
end

function (::G2P_Default)(pointstate, grid::Grid{dim, <: Any, <: WLS}, cache::MPCache{dim, <: Any, <: WLSValues}, dt::Real) where {dim}
    P = basis_function(grid.interpolation)
    p0 = value(P, zero(Vec{dim, Int}))
    ∇p0 = gradient(P, zero(Vec{dim, Int}))
    grid_to_point!(pointstate.C, cache) do mp, i, p
        @_inline_propagate_inbounds_meta
        w = mp.w
        M⁻¹ = mp.M⁻¹
        grid.state.v[i] ⊗ (w * M⁻¹ ⋅ value(P, grid[i] - pointstate.x[p]))
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

function (::G2P_Default)(pointstate, grid::Grid, cache::MPCache, dt::Real)
    G2P! = default_g2p(grid.interpolation)
    G2P!(pointstate, grid, cache, dt)
end
