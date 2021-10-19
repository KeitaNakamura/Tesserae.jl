struct Node
    f::Vec{2, Float64}
    w::Float64
    m::Float64
    v::Vec{2, Float64}
    v_n::Vec{2, Float64}
end

struct Point
    m::Float64
    V::Float64
    x::Vec{2, Float64}
    v::Vec{2, Float64}
    b::Vec{2, Float64}
    σ::SymmetricSecondOrderTensor{3, Float64, 6}
    F::SecondOrderTensor{3, Float64, 9}
    ∇v::SecondOrderTensor{3, Float64, 9}
    C::Mat{2, 3, Float64, 6}
end

@testset "P2G" begin
    for it in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
        for coordinate_system in (:plane_strain, :axisymmetric)
            # initialization
            grid = Grid(Node, it, 0.0:2.0:10.0, 0.0:2.0:20.0; coordinate_system)
            pointstate = generate_pointstate((x,y) -> true, Point, grid)
            cache = MPCache(grid, pointstate.x)
            v0 = rand(Vec{2})
            ρ0 = 1.2e3
            @. pointstate.m = ρ0 * pointstate.V
            @. pointstate.v = v0
            @. pointstate.σ = one(SymmetricSecondOrderTensor{3})
            @. pointstate.F = one(SecondOrderTensor{3})
            # transfer
            update!(cache, grid, pointstate.x)
            default_point_to_grid!(grid, pointstate, cache)
            @test all(==(v0), pointstate.v)
        end
    end
end
