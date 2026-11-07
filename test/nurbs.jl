nurbs_test_axes(degrees, knot_vectors) = map(Tesserae.NURBS.BSplineAxis, degrees, knot_vectors)
nurbs_test_control_net(degrees, knot_vectors, points, weights...) = Tesserae.NURBS.ControlNet(nurbs_test_axes(degrees, knot_vectors), points, weights...)
nurbs_test_degrees(net::Tesserae.NURBS.ControlNet) = map(axis -> axis.degree, net.axes)
nurbs_test_knot_vectors(net::Tesserae.NURBS.ControlNet) = map(axis -> axis.knot_vector, net.axes)

const nurbs_test_linear = Tesserae.NURBS.linear
const nurbs_test_quadratic = Tesserae.NURBS.quadratic
const nurbs_test_cubic = Tesserae.NURBS.cubic

function nurbs_sequential_insert(net, knots; direction)
    N = Tesserae.NURBS
    axis = net.axes[direction]
    points = N.homogeneous_points(net)
    for ξ in knots
        newaxis = N.insert_knot(axis, ξ)
        points = N.insert_knot_values(points, axis, ξ, direction)
        axis = newaxis
    end
    N.rational_control_net(Base.setindex(net.axes, axis, direction), points)
end

@testset "Knot insertion preserves cancellation and scalar arithmetic" begin
    N = Tesserae.NURBS
    for T in (Float16, Float32, Float64, BigFloat, Rational{Int})
        axis = N.BSplineAxis(2, T[0,0,0,1//4,1//2,3//4,1,1,1])
        points = Vec{2,T}[Vec(1,10), Vec(4,-7), Vec(-6,3), Vec(6,-9), Vec(-7,-6), Vec(-7,-5)]
        weight_sets = [T[1,1,-1,1,2,2], T[0,1,1,1,2,2]]
        T <: AbstractFloat && push!(weight_sets, fill(nextfloat(zero(T)), 6))
        T <: AbstractFloat && append!(weight_sets, [T[Inf,1,1,1,2,2], T[NaN,1,1,1,2,2]])
        T <: AbstractFloat || push!(weight_sets, T[1,2,3,4,5,6])
        outcome(f) = try f() catch err; typeof(err) end
        for weights in weight_sets
            net = N.ControlNet((axis,), points, weights)
            knots = T[1//8,1//4,5//8]
            expected = outcome(() -> nurbs_sequential_insert(net, knots; direction=1))
            target = N.insert_knot(axis, knots)
            for actual in (outcome(() -> N.insert_knot(net, knots; direction=1)),
                           outcome(() -> N.refineto(net, target; direction=1)))
                @test typeof(actual) === typeof(expected)
                if actual isa N.ControlNet && expected isa N.ControlNet
                    @test isequal(actual.points, expected.points)
                    @test isequal(actual.weights, expected.weights)
                    if T === Float64 && weights == [1,1,-1,1,2,2]
                        @test all(point -> all(isfinite, point), actual.points)
                    end
                else
                    @test actual === expected
                end
            end
            uniform_knots = T[1//8,3//8,5//8,7//8]
            expected_uniform = outcome(() -> nurbs_sequential_insert(net, uniform_knots; direction=1))
            actual_uniform = outcome(() -> N.refine(net, 1; direction=1))
            @test typeof(actual_uniform) === typeof(expected_uniform)
            if actual_uniform isa N.ControlNet && expected_uniform isa N.ControlNet
                @test isequal(actual_uniform.points, expected_uniform.points)
                @test isequal(actual_uniform.weights, expected_uniform.weights)
            end
        end
        net = N.ControlNet((axis,), points, T[1,2,3,4,5,6])
        @test N.insert_knot(net, T[]; direction=1).points == N.rational_control_net((axis,),N.homogeneous_points(net)).points
    end
    for T in (Float32, Float64)
        limit = floatmax(T)
        axis = N.BSplineAxis(1, T[-limit,-limit,limit,limit])
        net = N.ControlNet((axis,), [Vec(one(T)), Vec(T(2))])
        knots = T[-limit/2, -limit/4]
        expected = nurbs_sequential_insert(net, knots; direction=1)
        for actual in (N.insert_knot(net, knots; direction=1),
                       N.refineto(net, N.insert_knot(axis, knots); direction=1))
            @test all(point -> all(isfinite, point), actual.points)
            @test actual.points == expected.points
            @test actual.weights == expected.weights
        end
    end
end

@testset "Knot insertion retains small endpoint contributions" begin
    N = Tesserae.NURBS
    for T in (Float16, Float32, Float64), δ in (eps(T)/16,3eps(T)/4), degree in 1:5, side in (-1,1), pdim in 1:3, direction in 1:pdim
        ends = side == 1 ? (zero(T),one(T)) : (-one(T),zero(T))
        axis = N.BSplineAxis(degree, vcat(fill(ends[1],degree+1), fill(ends[2],degree+1)))
        net_axes = ntuple(d -> d == direction ? axis : N.open_bspline_axis(T, 1, 1), pdim)
        points = map(CartesianIndices(map(N.nbasis, net_axes))) do I
            fraction = T(I[direction]-1) / T(degree)
            Vec((side == 1 ? fraction : 1-fraction) / δ)
        end
        net = N.ControlNet(net_axes, points)
        ξ = T(side) * δ
        knots = sort(T[ξ, T(side)/2])
        coordinate = Vec(ntuple(d -> d == direction ? ξ : T(1//4), pdim))
        # Both curves have the analytic value 1 here. Merely comparing the bulk
        # and single paths would miss complementary rounding errors in either.
        @test N.evaluate(net, coordinate) ≈ Vec(one(T))
        for refined in (N.insert_knot(net, ξ; direction),
                        N.insert_knot(net, knots; direction),
                        N.refineto(net, N.insert_knot(axis, ξ); direction),
                        N.refineto(net, N.insert_knot(axis, knots); direction))
            @test N.evaluate(refined, coordinate) ≈ Vec(one(T))
            @test all(p -> all(isfinite, p), refined.points)
        end
        for piece in split(net, direction, ξ)
            @test N.evaluate(piece, coordinate) ≈ Vec(one(T))
        end
    end
end

@testset "Knot insertion retains small interior-span contributions" begin
    N = Tesserae.NURBS
    for T in (Float32, Float64), side in (-1,1)
        a = T(2)^(-10)
        δ = nextfloat(a) - a
        ξ = T(side) * nextfloat(a)
        kv = side == 1 ? T[0,0,a,1,1] : T[-1,-1,-a,0,0]
        points = side == 1 ? [Vec(zero(T)),Vec(zero(T)),Vec(inv(δ))] :
                             [Vec(inv(δ)),Vec(zero(T)),Vec(zero(T))]
        net = N.ControlNet((N.BSplineAxis(1,kv),), points)
        knots = sort(T[ξ,T(side)/2])
        expected = Vec(inv(one(T)-a))
        @test N.evaluate(net, Vec(ξ)) ≈ expected
        for refined in (N.insert_knot(net, ξ; direction=1),
                        N.insert_knot(net, knots; direction=1),
                        N.refineto(net, N.insert_knot(net.axes[1],knots); direction=1))
            @test N.evaluate(refined, Vec(ξ)) ≈ expected
        end
    end
end

@testset "BigFloat small endpoint contributions" begin
    N = Tesserae.NURBS
    for side in (-1,1), δ in (eps(BigFloat)/16,3eps(BigFloat)/4)
        ends = side == 1 ? BigFloat[0,0,1,1] : BigFloat[-1,-1,0,0]
        points = side == 1 ? [Vec(BigFloat(0)),Vec(inv(δ))] : [Vec(inv(δ)),Vec(BigFloat(0))]
        net = N.ControlNet((N.BSplineAxis(1,ends),),points)
        ξ = side*δ
        for refined in (N.insert_knot(net,ξ;direction=1),N.insert_knot(net,sort(BigFloat[ξ,side//2]);direction=1))
            @test N.evaluate(refined,Vec(ξ)) ≈ Vec(BigFloat(1))
        end
    end
end

@testset "Knot insertion coefficients retain relative accuracy" begin
    N = Tesserae.NURBS
    # Both coefficients fit Rational{Int64}, although right-ξ does not.
    b, d = Int64(4_000_000_000), Int64(4_000_000_001)
    z, o = zero(Rational{Int64}), one(Rational{Int64})
    right, ξ = 1//b, 1//d
    @test N.insertion_coefficients(z,ξ,right) == (b//d,1//d)
    axis = N.BSplineAxis(1,[z,z,right,right])
    net = N.ControlNet((axis,),[Vec(z),Vec(o)])
    @test N.insert_knot(net,ξ;direction=1).points == [Vec(z),Vec(b//d),Vec(o)]
    for T in (Float16,Float32,Float64), scale in (one(T),floatmin(T),64nextfloat(zero(T)),floatmax(T)/4),
        δ in (eps(T)/16,3eps(T)/4,16eps(T),sqrt(eps(T)),T(1//100),T(1//3))
        for (left,ξ,right) in ((zero(T),scale*δ,scale),(-scale,-scale*δ,zero(T)),
                              (scale,nextfloat(scale),2scale))
            left < ξ < right || continue
            α, β = N.insertion_coefficients(left,ξ,right)
            # Exact rational arithmetic is independent of both insertion paths.
            l, x, r = Rational{BigInt}.((left,ξ,right))
            @test isapprox(α,T((x-l)/(r-l));rtol=8eps(T),atol=0)
            @test isapprox(β,T((r-x)/(r-l));rtol=8eps(T),atol=0)
            @test α + β == one(T)
        end
    end
    for T in (Float16,Float32,Float64,BigFloat),
        δ in (16eps(T),sqrt(eps(T)),T(1//10000),T(1//100),T(1//3)), side in (-1,1)
        ends = side == 1 ? T[0,0,1,1] : T[-1,-1,0,0]
        points = side == 1 ? [Vec(zero(T)),Vec(inv(δ))] : [Vec(inv(δ)),Vec(zero(T))]
        net = N.ControlNet((N.BSplineAxis(1,ends),),points)
        ξ = side*δ
        knots = sort(T[ξ,side//2])
        target = N.insert_knot(net.axes[1],knots)
        for refined in (N.insert_knot(net,ξ;direction=1),N.insert_knot(net,knots;direction=1),
                        N.refineto(net,target;direction=1),split(net,1,ξ)...)
            @test isapprox(N.evaluate(refined,Vec(ξ)),Vec(one(T));rtol=8eps(T),atol=0)
        end
    end
    for precision in (64,128,256)
        setprecision(BigFloat,precision) do
            for δ in (sqrt(eps(BigFloat)),16eps(BigFloat),BigFloat(1//10000))
                α,β = N.insertion_coefficients(BigFloat(-1),-δ,BigFloat(0))
                @test isapprox(β,δ;rtol=8eps(BigFloat),atol=0)
                @test α + β == 1
            end
        end
    end
end

@testset "Knot insertion paths preserve geometry and input order" begin
    N = Tesserae.NURBS
    for T in (Float32, Float64), pdim in 1:3, direction in 1:pdim
        axes = ntuple(_ -> N.open_bspline_axis(T, 3, 2), pdim)
        points = [Vec{3,T}(ntuple(d -> sin(T(sum(Tuple(I)) + d)), 3)) for I in CartesianIndices(map(N.nbasis, axes))]
        weights = [T(1 + sum(Tuple(I)) / 7) for I in CartesianIndices(points)]
        net = N.ControlNet(axes, points, weights)
        for knots in (T[], T[0.3], T[0.2,0.2,0.5,0.7], T[0.7,0.2,0.5,0.2])
            sequential = net
            for ξ in knots
                sequential = N.insert_knot(sequential, ξ; direction)
            end
            inserted = N.insert_knot(net, view(knots, :); direction)
            # Identity ranges preserve a non-one-based vector axis without an
            # extra dependency. Public knot insertion accepts AbstractVector.
            shifted = view(vcat(zero(T), knots), Base.IdentityUnitRange(2:length(knots)+1))
            from_shifted = N.insert_knot(net, shifted; direction)
            @test from_shifted.points == inserted.points
            @test from_shifted.weights == inserted.weights
            @test inserted.axes == sequential.axes
            @test inserted.points ≈ sequential.points
            @test inserted.weights ≈ sequential.weights
            # Unsorted input must take precisely the requested operation order,
            # including keeping homogeneous coordinates between insertions.
            if !issorted(knots)
                hp, axis = N.homogeneous_points(net), axes[direction]
                for ξ in knots
                    hp = N.insert_knot_values(hp, axis, ξ, direction)
                    axis = N.insert_knot(axis, ξ)
                end
                ordered = N.rational_control_net(Base.setindex(axes, axis, direction), hp)
                @test inserted.points == ordered.points
                @test inserted.weights == ordered.weights
            end
            target = N.refineto(net, inserted.axes[direction]; direction)
            @test target.axes == inserted.axes
            @test target.points ≈ sequential.points
            @test target.weights ≈ sequential.weights
            for u in (zero(T), T(0.17), T(0.5), T(0.89), one(T))
                ξ = Vec(ntuple(_ -> u, pdim))
                @test N.evaluate(inserted, ξ) ≈ N.evaluate(net, ξ)
                @test N.evaluate(target, ξ) ≈ N.evaluate(net, ξ)
            end
        end
        @test_throws ArgumentError N.insert_knot(net, T[0.2, 0]; direction)
        @test_throws ArgumentError N.insert_knot(net, fill(T(0.2), 4); direction)
    end
end

@testset "Refined rational geometry agrees with exact arithmetic" begin
    N = Tesserae.NURBS
    R = Rational{BigInt}
    for T in (Float32,Float64), p in 1:5, side in (-1,1),
        δ in (16eps(T),sqrt(eps(T)),T(1//10000),T(1//100)), weight_mode in (:unit,:positive,:signed)
        ends = side == 1 ? (zero(T),one(T)) : (-one(T),zero(T))
        axis = N.BSplineAxis(p,vcat(fill(ends[1],p+1),fill(ends[2],p+1)))
        points = [Vec(T(i)/δ,one(T)+T(i)) for i in 0:p]
        side == -1 && reverse!(points)
        weights = weight_mode == :positive ? T[1+i//3 for i in 0:p] : ones(T,p+1)
        if weight_mode == :signed
            weights[side == 1 ? 1 : end] = -T(p)*δ/2
        end
        net = N.ControlNet((axis,),points,weights)
        exact = N.ControlNet((N.BSplineAxis(p,R.(axis.knot_vector)),),Vec{2,R}.(points),R.(weights))
        ξ = T(side)*δ
        expected = Vec{2,T}(N.evaluate(exact,Vec(R(ξ))))
        knots = sort(T[ξ,T(side)/2])
        target = N.insert_knot(axis,knots)
        for refined in (N.insert_knot(net,ξ;direction=1),N.insert_knot(net,knots;direction=1),
                        N.refineto(net,target;direction=1))
            @test isapprox(N.evaluate(refined,Vec(ξ)),expected;rtol=32eps(T),atol=0)
        end
    end
end

@testset "Refinement of asymmetric tensor-product control nets" begin
    N = Tesserae.NURBS
    for T in (Float32,Float64), direction in (1,3), knots in (T[0.013,0.347,0.973], T.(collect(range(0.013,0.973;length=65))[1:64]))
        net_axes = map(n -> N.open_bspline_axis(T,3,n), (14,16,18))
        points = [Vec(sin(T(I[1]+2I[2]+3I[3])),cos(T(3I[1]-I[2]+4I[3])),T(I[1]*I[2]+2I[3])/17)
                  for I in CartesianIndices(map(N.nbasis,net_axes))]
        weights = [one(T)+T(I[1]+3I[2]+5I[3])/31 for I in CartesianIndices(points)]
        net = N.ControlNet(net_axes,points,weights)
        inserted = N.insert_knot(net,knots;direction)
        expected = nurbs_sequential_insert(net,knots;direction)
        @test inserted.points ≈ expected.points
        @test inserted.weights ≈ expected.weights
        targeted = N.refineto(net,inserted.axes[direction];direction)
        @test targeted.points ≈ expected.points
        @test targeted.weights ≈ expected.weights
        for ξ in (Vec(T(0.17),T(0.41),T(0.83)),Vec(T(0.78),T(0.23),T(0.56)))
            @test N.evaluate(inserted,ξ) ≈ N.evaluate(net,ξ)
        end
    end
end

@testset "Cached refinement coefficients agree with direct evaluation" begin
    N = Tesserae.NURBS
    for T in (Float32,Float64), p in (1,3,5,10), nfibers in (0,1,16,17,21,64), direction in (1,2)
        axis = N.open_bspline_axis(T,p,3)
        knots = p == 1 ? T[0.13,0.61,0.87] : T[0.13,0.13,0.61,0.87]
        target = N.insert_knot(axis,knots)
        dims = direction == 1 ? (N.nbasis(axis),nfibers) : (nfibers,N.nbasis(axis))
        values = Vec{2,T}[Vec(T(1)+T(I[1])/7,T(2)+T(I[2])/13) for I in CartesianIndices(dims)]
        actual = N.refine_values(values,axis,target,knots,direction)
        direct = N.refine_values(values,axis,target,knots,direction,nothing)
        @test actual == direct
        sequential, current_axis = values, axis
        for ξ in knots
            sequential = N.insert_knot_values(sequential,current_axis,ξ,direction)
            current_axis = N.insert_knot(current_axis,ξ)
        end
        @test isapprox(actual,sequential;rtol=32eps(T),atol=0)
    end
end

@testset "Degree elevation across unequal knot multiplicities" begin
    N = Tesserae.NURBS
    R = Rational{BigInt}
    axis = N.BSplineAxis(2, R[0,0,0,1//4,1//2,1//2,1,1,1])
    other = N.BSplineAxis(1, R[0,0,1//2,1,1])
    points = [Vec(R(i*i+j), R(i-j*j)) for i in 1:3, j in 1:6]
    weights = [one(R) + R(i+2j)/13 for i in 1:3, j in 1:6]
    net = N.ControlNet((other, axis), points, weights)
    elevated = N.elevate(net, 5; direction=2)
    @test elevated.axes[2].knot_vector == vcat(fill(R(0),6), fill(R(1//4),4), fill(R(1//2),5), fill(R(1),6))
    for (v,u) in ((0,0),(2//5,1//8),(3//4,1//4),(1,1//2),(1//5,3//4),(4//5,1))
        ξ = Vec(R(v), R(u))
        @test N.evaluate(elevated, ξ) == N.evaluate(net, ξ)
    end
    constant = N.ControlNet((N.BSplineAxis(0, R[0,1]),), [Vec(R(7//3))], R[2//5])
    elevated_constant = N.elevate(constant, 3; direction=1)
    @test (elevated_constant.points, elevated_constant.weights) == (fill(Vec(R(7//3)),4), fill(R(2//5),4))
    invalid = N.ControlNet((N.BSplineAxis(0, [0.0,NaN]),), [Vec(1.0)])
    @test_throws ArgumentError N.elevate(invalid, 2; direction=1)
end

@testset "NURBS" begin
    # Shared fixtures: one rational quadratic curve and one tensor-product
    # surface used to check geometry-preserving operations.
    curve_degrees = (nurbs_test_quadratic,)
    curve_knots = ([0.0,0.0,0.0,0.5,1.0,1.0,1.0],)
    curve_controlpoints = [Vec(Float64(i-1), 0.0) for i in 1:4]
    curve_weights = collect(1.0:4.0)
    curve_control = nurbs_test_control_net(curve_degrees, curve_knots, curve_controlpoints, curve_weights)

    surface_degrees = (nurbs_test_quadratic, nurbs_test_quadratic)
    surface_knots = ([0.0,0.0,0.0,0.5,1.0,1.0,1.0], [0.0,0.0,0.0,1/3,2/3,1.0,1.0,1.0])
    surface_controlpoints = map(CartesianIndices((4, 5))) do I
        Vec(Float64(I[1]-1), Float64(I[2]-1))
    end
    surface_control = nurbs_test_control_net(surface_degrees, surface_knots, surface_controlpoints)

    @testset "Primitives" begin
        # Linear primitives should expose the expected axis metadata and
        # evaluate to affine interpolation.
        line_curve = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(2.0, 4.0))
        @test nurbs_test_degrees(line_curve) == (nurbs_test_linear,)
        @test nurbs_test_knot_vectors(line_curve) == ([0.0,0.0,1.0,1.0],)
        @test Tesserae.NURBS.degree(line_curve.axes[1]) == nurbs_test_linear
        @test Tesserae.NURBS.degree(line_curve, 1) == nurbs_test_linear
        @test Tesserae.NURBS.knots(line_curve.axes[1]) == [0.0,0.0,1.0,1.0]
        @test Tesserae.NURBS.knots(line_curve, 1) == [0.0,0.0,1.0,1.0]
        @test Tesserae.NURBS.domain(line_curve.axes[1]) == (0.0, 1.0)
        @test Tesserae.NURBS.domain(line_curve, 1) == (0.0, 1.0)
        @test_throws ArgumentError Tesserae.NURBS.degree(line_curve, 2)
        @test_throws ArgumentError Tesserae.NURBS.knots(line_curve, 2)
        @test_throws ArgumentError Tesserae.NURBS.domain(line_curve, 2)
        @test line_curve.points == [Vec(0.0, 0.0), Vec(2.0, 4.0)]
        @test (@inferred Tesserae.NURBS.evaluate(line_curve, Vec(0.25))) ≈ Vec(0.5, 1.0)

        polyline_curve = Tesserae.NURBS.polyline([Vec(0.0, 0.0), Vec(1.0, 2.0), Vec(3.0, 3.0)])
        @test nurbs_test_degrees(polyline_curve) == (nurbs_test_linear,)
        @test nurbs_test_knot_vectors(polyline_curve) == ([0.0,0.0,0.5,1.0,1.0],)
        @test polyline_curve.points == [Vec(0.0, 0.0), Vec(1.0, 2.0), Vec(3.0, 3.0)]
        @test_throws ArgumentError Tesserae.NURBS.polyline([Vec(0.0, 0.0)])

        # Circular arcs are exact rational quadratic curves; the midpoint check
        # catches both control-point placement and rational weights.
        unit_arc = Tesserae.NURBS.arcunit(Float32(π/2))
        @test nurbs_test_degrees(unit_arc) == (nurbs_test_quadratic,)
        @test eltype(unit_arc.axes[1].knot_vector) === Float32
        @test eltype(unit_arc.weights) === Float32
        @test unit_arc.points[1] ≈ Vec(1.0f0, 0.0f0)
        @test unit_arc.points[2] ≈ Vec(1.0f0, 1.0f0)
        @test unit_arc.points[3] ≈ Vec(0.0f0, 1.0f0)
        @test Tesserae.NURBS.evaluate(unit_arc, Vec(0.5f0)) ≈ Vec(sqrt(0.5f0), sqrt(0.5f0))

        generated_arc = Tesserae.NURBS.arc(Vec(0.0, 0.0), 1.0, 0.0, π/2)
        @test nurbs_test_degrees(generated_arc) == (nurbs_test_quadratic,)
        @test nurbs_test_knot_vectors(generated_arc) == ([0.0,0.0,0.0,1.0,1.0,1.0],)
        @test generated_arc.points[1] ≈ Vec(1.0, 0.0)
        @test generated_arc.points[2] ≈ Vec(1.0, 1.0)
        @test generated_arc.points[3] ≈ Vec(0.0, 1.0)
        @test generated_arc.weights ≈ [1.0, √2/2, 1.0]
        @test Tesserae.NURBS.evaluate(generated_arc, Vec(0.5)) ≈ Vec(√2/2, √2/2)
        @test_throws ArgumentError Tesserae.NURBS.arc(Vec(0.0, 0.0), 0.0, 0.0, π/2)
        @test_throws ArgumentError Tesserae.NURBS.arc(Vec(0.0, 0.0), 1.0, 0.0, 0.0)

        # A full circle is represented as four rational quadratic arc segments.
        generated_circle = Tesserae.NURBS.circle(Vec(0.0, 0.0), 1.0)
        @test nurbs_test_degrees(generated_circle) == (nurbs_test_quadratic,)
        @test nurbs_test_knot_vectors(generated_circle) == ([0.0,0.0,0.0,0.25,0.25,0.5,0.5,0.75,0.75,1.0,1.0,1.0],)
        @test generated_circle.points[begin] ≈ Vec(1.0, 0.0)
        @test generated_circle.points[end] ≈ Vec(1.0, 0.0)
        @test generated_circle.weights ≈ [1.0, √2/2, 1.0, √2/2, 1.0, √2/2, 1.0, √2/2, 1.0]

        shifted_arc = Tesserae.NURBS.arc(Vec(0.0, 0.0), 1.0, π/2, π)
        @test shifted_arc.points[1] ≈ Vec(0.0, 1.0)
        @test shifted_arc.points[2] ≈ Vec(-1.0, 1.0)
        @test shifted_arc.points[3] ≈ Vec(-1.0, 0.0)

        generated_arc3d = Tesserae.NURBS.arc(Vec(0.0, 0.0, 1.0), 2.0, 0.0, π/2; normal=Vec(0.0, 0.0, 1.0))
        @test generated_arc3d.points[1] ≈ Vec(2.0, 0.0, 1.0)
        @test generated_arc3d.points[2] ≈ Vec(2.0, 2.0, 1.0)
        @test generated_arc3d.points[3] ≈ Vec(0.0, 2.0, 1.0)
        @test Tesserae.NURBS.evaluate(generated_arc3d, Vec(0.5)) ≈ Vec(√2, √2, 1.0)
        @test Tesserae.NURBS.default_arc_xaxis(Vec(0.0, 1.0, 0.0)) ≈ Vec(-1.0, 0.0, 0.0)

        # Reversing a curve should only reverse the parameterization, not move
        # the represented geometry.
        reversed_arc = reverse(generated_arc)
        @test reversed_arc.points[begin] ≈ generated_arc.points[end]
        @test reversed_arc.points[end] ≈ generated_arc.points[begin]
        @test Tesserae.NURBS.evaluate(reversed_arc, Vec(0.25)) ≈ Tesserae.NURBS.evaluate(generated_arc, Vec(0.75))
    end

    @testset "Surfaces" begin
        # Coons patches blend four boundary curves into one tensor-product
        # surface with the merged boundary bases.
        bottom = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(2.0, 0.0))
        right = Tesserae.NURBS.line(Vec(2.0, 0.0), Vec(3.0, 1.0))
        top = Tesserae.NURBS.line(Vec(0.0, 1.0), Vec(3.0, 1.0))
        left = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(0.0, 1.0))
        coons_quad = Tesserae.NURBS.coons_patch(bottom, top, left, right)
        @test nurbs_test_degrees(coons_quad) == (nurbs_test_linear, nurbs_test_linear)
        @test size(coons_quad.points) == (2, 2)
        @test (@inferred Tesserae.NURBS.evaluate(coons_quad, Vec(0.5, 0.5))) ≈ Vec(1.25, 0.5)

        # Curved boundaries should remain exact after Coons blending; this
        # annular sector checks the radial midpoint and angular midpoint.
        inner = Tesserae.NURBS.arc(Vec(0.0, 0.0), 1.0, 0.0, π/2)
        outer = Tesserae.NURBS.arc(Vec(0.0, 0.0), 2.0, 0.0, π/2)
        radial0 = Tesserae.NURBS.line(Vec(1.0, 0.0), Vec(2.0, 0.0))
        radial1 = Tesserae.NURBS.line(Vec(0.0, 1.0), Vec(0.0, 2.0))
        annular_sector = Tesserae.NURBS.coons_patch(inner, outer, radial0, radial1)
        @test nurbs_test_degrees(annular_sector) == (nurbs_test_quadratic, nurbs_test_linear)
        @test size(annular_sector.points) == (3, 2)
        @test Tesserae.NURBS.evaluate(annular_sector, Vec(0.0, 0.5)) ≈ Vec(1.5, 0.0)
        @test Tesserae.NURBS.evaluate(annular_sector, Vec(1.0, 0.5)) ≈ Vec(0.0, 1.5)
        @test Tesserae.NURBS.evaluate(annular_sector, Vec(0.5, 0.5)) ≈ Vec(1.5 / √2, 1.5 / √2)
        @test_throws ArgumentError Tesserae.NURBS.coons_patch(bottom, top, right, left)

        # Coons blending first moves all four boundary curves to common bases.
        # Here the bottom curve already has an interior knot, so the top curve
        # must be refined to the same first-direction axis.
        kinked_bottom = Tesserae.NURBS.polyline([Vec(0.0, 0.0), Vec(0.5, 0.0), Vec(1.0, 0.0)])
        straight_top = Tesserae.NURBS.line(Vec(0.0, 1.0), Vec(1.0, 1.0))
        straight_left = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(0.0, 1.0))
        straight_right = Tesserae.NURBS.line(Vec(1.0, 0.0), Vec(1.0, 1.0))
        refined_coons = Tesserae.NURBS.coons_patch(kinked_bottom, straight_top, straight_left, straight_right)
        @test nurbs_test_knot_vectors(refined_coons)[1] == [0.0,0.0,0.5,1.0,1.0]
        @test Tesserae.NURBS.evaluate(refined_coons, Vec(0.5, 0.5)) ≈ Vec(0.5, 0.5)

        @test surface_control isa Tesserae.NURBS.ControlNet
        @test size(surface_control.points) == (4, 5)
        @test all(isone, surface_control.weights)
        @test surface_control.points[1,1] ≈ Vec(0.0, 0.0)
        @test surface_control.points[end,1] ≈ Vec(3.0, 0.0)
        @test surface_control.points[end,end] ≈ Vec(3.0, 4.0)
        @test surface_control.points[1,end] ≈ Vec(0.0, 4.0)

        # Boundary extraction drops the fixed parametric direction while
        # preserving the remaining axis and boundary control points.
        surface_boundaries = Tesserae.NURBS.boundaries(surface_control)
        @test length(surface_boundaries) == 4
        @test surface_boundaries[1].points == surface_controlpoints[1,:]
        @test surface_boundaries[2].points == surface_controlpoints[end,:]
        @test surface_boundaries[3].points == surface_controlpoints[:,1]
        @test surface_boundaries[4].points == surface_controlpoints[:,end]
        @test nurbs_test_degrees(surface_boundaries[1]) == (nurbs_test_quadratic,)
        @test nurbs_test_knot_vectors(surface_boundaries[1]) == (surface_knots[2],)
        @test_throws ArgumentError Tesserae.NURBS.boundaries(surface_control, 3, -1)
        @test_throws ArgumentError Tesserae.NURBS.boundaries(surface_control, 1, 0)
    end

    @testset "Basis utilities" begin
        # Dense basis matrices are filled only on the active basis-function
        # columns, but each row still forms a partition of unity.
        axis = Tesserae.NURBS.BSplineAxis(nurbs_test_linear, [0.0,0.0,0.5,1.0,1.0])
        basis_values = Tesserae.NURBS.basis_matrix(axis, [0.0, 0.25, 0.5, 0.75, 1.0])
        @test size(basis_values) == (5, 3)
        @test vec(sum(basis_values; dims=2)) ≈ ones(5)
        @test basis_values[1,:] ≈ [1.0, 0.0, 0.0]
        @test basis_values[end,:] ≈ [0.0, 0.0, 1.0]

        # Tuple-index helpers should reject impossible parametric directions.
        @test Tesserae.NURBS.dropat((:u, :v, :w), 2) == (:u, :w)
        @test_throws ArgumentError Tesserae.NURBS.dropat((:u, :v), 3)
    end

    @testset "Refinement" begin
        # Knot insertion should add basis functions without changing the curve.
        # The weighted control points are checked through the rational weights.
        refined_curve = Tesserae.NURBS.insert_knot(curve_control, 0.25; direction=1)
        @test nurbs_test_knot_vectors(refined_curve) == ([0.0,0.0,0.0,0.25,0.5,1.0,1.0,1.0],)
        @test refined_curve.weights ≈ [1.0, 1.5, 2.25, 3.0, 4.0]
        @test refined_curve.points[1] ≈ Vec(0.0, 0.0)
        @test refined_curve.points[2] ≈ Vec(2/3, 0.0)
        @test refined_curve.points[3] ≈ Vec(4/3, 0.0)
        @test refined_curve.points[4] ≈ Vec(2.0, 0.0)
        @test refined_curve.points[5] ≈ Vec(3.0, 0.0)
        @test_throws ArgumentError Tesserae.NURBS.insert_knot(curve_control, 0.25; direction=2)
        @test_throws ArgumentError Tesserae.NURBS.insert_knot(curve_control, 0.0; direction=1)
        @test_throws ArgumentError Tesserae.NURBS.insert_knot(curve_control, 0.25; direction=1, ntimes=3)
        @test_throws MethodError Tesserae.NURBS.insert_knot(curve_control, 1, 0.25)

        # Vector insertion and uniform refinement should land on the same knot
        # vector here and preserve point evaluation.
        vector_refined_curve = Tesserae.NURBS.insert_knot(curve_control, [0.25, 0.75]; direction=1)
        @test nurbs_test_knot_vectors(vector_refined_curve) == ([0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0],)
        for ξ in (0.1, 0.4, 0.8)
            @test Tesserae.NURBS.evaluate(vector_refined_curve, Vec(ξ)) ≈ Tesserae.NURBS.evaluate(curve_control, Vec(ξ))
        end

        direction_refined_curve = Tesserae.NURBS.refine(curve_control, 1; direction=1)
        @test nurbs_test_knot_vectors(direction_refined_curve) == nurbs_test_knot_vectors(vector_refined_curve)
        tuple_refined_curve = Tesserae.NURBS.refine(curve_control, (1,))
        @test nurbs_test_knot_vectors(tuple_refined_curve) == nurbs_test_knot_vectors(vector_refined_curve)
        @test_throws ArgumentError Tesserae.NURBS.refine(curve_control, -1; direction=1)

        # Surface knot insertion is checked in the second parametric direction
        # so the directional array conversion path is exercised.
        refined_surface = Tesserae.NURBS.insert_knot(surface_control, 0.5; direction=2)
        @test size(refined_surface.points) == (4, 6)
        @test nurbs_test_knot_vectors(refined_surface)[1] == surface_knots[1]
        @test nurbs_test_knot_vectors(refined_surface)[2] == [0.0,0.0,0.0,1/3,0.5,2/3,1.0,1.0,1.0]
        @test refined_surface.points[2,1] ≈ Vec(1.0, 0.0)
        @test refined_surface.points[2,3] ≈ Vec(1.0, 1.75)
        @test refined_surface.points[2,4] ≈ Vec(1.0, 2.25)
        @test refined_surface.points[2,end] ≈ Vec(1.0, 4.0)
        @test all(isone, refined_surface.weights)

        twice_refined_surface = Tesserae.NURBS.insert_knot(surface_control, 0.5; direction=2, ntimes=2)
        @test size(twice_refined_surface.points) == (4, 7)
        @test nurbs_test_knot_vectors(twice_refined_surface)[2] == [0.0,0.0,0.0,1/3,0.5,0.5,2/3,1.0,1.0,1.0]

        # Refining directly to a target axis handles repeated inserted knots in
        # one pass and should preserve the geometry.
        target_axis = Tesserae.NURBS.BSplineAxis(nurbs_test_quadratic, [0.0,0.0,0.0,0.25,0.25,0.5,1.0,1.0,1.0])
        target_refined_curve = Tesserae.NURBS.refineto(curve_control, target_axis; direction=1)
        @test nurbs_test_knot_vectors(target_refined_curve) == (target_axis.knot_vector,)
        for ξ in (0.1, 0.25, 0.4, 0.8)
            @test Tesserae.NURBS.evaluate(target_refined_curve, Vec(ξ)) ≈ Tesserae.NURBS.evaluate(curve_control, Vec(ξ))
        end

        direction_uniform_surface = Tesserae.NURBS.refine(surface_control, 1; direction=2)
        @test nurbs_test_knot_vectors(direction_uniform_surface)[1] == surface_knots[1]
        @test nurbs_test_knot_vectors(direction_uniform_surface)[2] ≈ [0.0,0.0,0.0,1/6,1/3,0.5,2/3,5/6,1.0,1.0,1.0]
        tuple_uniform_surface = Tesserae.NURBS.refine(surface_control, (1, 0))
        @test nurbs_test_knot_vectors(tuple_uniform_surface)[1] == [0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0]
        @test nurbs_test_knot_vectors(tuple_uniform_surface)[2] == surface_knots[2]
    end

    @testset "Elevation and split" begin
        # Degree elevation should raise the basis degree while preserving the
        # exact rational arc.
        circular_arc = nurbs_test_control_net(
            (nurbs_test_quadratic,),
            ([0.0,0.0,0.0,1.0,1.0,1.0],),
            [Vec(1.0,0.0), Vec(1.0,1.0), Vec(0.0,1.0)],
            [1.0, √2/2, 1.0],
        )
        elevated_arc = Tesserae.NURBS.elevate(circular_arc, nurbs_test_cubic; direction=1)
        @test nurbs_test_degrees(elevated_arc) == (nurbs_test_cubic,)
        @test nurbs_test_knot_vectors(elevated_arc) == ([0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0],)
        @test size(elevated_arc.points) == (4,)
        elevated_arc_by_ntimes = @inferred Tesserae.NURBS.elevate(circular_arc; direction=1)
        @test nurbs_test_degrees(elevated_arc_by_ntimes) == (nurbs_test_cubic,)
        @test nurbs_test_knot_vectors(elevated_arc_by_ntimes) == nurbs_test_knot_vectors(elevated_arc)
        for ξ in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0)
            @test Tesserae.NURBS.evaluate(elevated_arc, Vec(ξ)) ≈ Tesserae.NURBS.evaluate(circular_arc, Vec(ξ))
        end

        refined_elevated_arc = Tesserae.NURBS.insert_knot(elevated_arc, [0.25, 0.5, 0.75]; direction=1)
        @test nurbs_test_degrees(refined_elevated_arc) == (nurbs_test_cubic,)
        @test nurbs_test_knot_vectors(refined_elevated_arc) == ([0.0,0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0,1.0],)
        for ξ in (0.0, 0.125, 0.375, 0.625, 0.875, 1.0)
            @test Tesserae.NURBS.evaluate(refined_elevated_arc, Vec(ξ)) ≈ Tesserae.NURBS.evaluate(circular_arc, Vec(ξ))
        end
        @test_throws ArgumentError Tesserae.NURBS.elevate(circular_arc, nurbs_test_cubic; direction=2)
        @test_throws ArgumentError Tesserae.NURBS.elevate(circular_arc, nurbs_test_linear; direction=1)
        @test_throws ArgumentError Tesserae.NURBS.elevate(circular_arc; direction=1, ntimes=-1)

        # The elevation implementation is intentionally limited to open,
        # non-discontinuous knot vectors.
        nonopen_curve = nurbs_test_control_net(
            (nurbs_test_quadratic,),
            ([0.0,0.0,0.5,1.0,1.0],),
            [Vec(0.0,0.0), Vec(1.0,0.0)],
        )
        @test_throws ArgumentError Tesserae.NURBS.elevate(nonopen_curve, nurbs_test_cubic; direction=1)
        discontinuous_curve = nurbs_test_control_net(
            (nurbs_test_quadratic,),
            ([0.0,0.0,0.0,0.5,0.5,0.5,1.0,1.0,1.0],),
            [Vec(Float64(i), 0.0) for i in 1:6],
        )
        @test_throws ArgumentError Tesserae.NURBS.elevate(discontinuous_curve, nurbs_test_cubic; direction=1)

        # Splitting should return two exact pieces that meet at the cut point.
        left_curve, right_curve = Tesserae.NURBS.split(curve_control, 1, 0.5)
        @test nurbs_test_knot_vectors(left_curve) == ([0.0,0.0,0.0,0.5,0.5,0.5],)
        @test nurbs_test_knot_vectors(right_curve) == ([0.5,0.5,0.5,1.0,1.0,1.0],)
        @test left_curve.weights ≈ [1.0, 2.0, 2.5]
        @test right_curve.weights ≈ [2.5, 3.0, 4.0]
        @test left_curve.points[end] ≈ right_curve.points[1]

        # Surface elevation and later refinement should still represent the
        # same map at sample parametric points.
        elevated_surface = @inferred Tesserae.NURBS.elevate(surface_control, (nurbs_test_cubic, nurbs_test_cubic))
        refined_elevated_surface = Tesserae.NURBS.insert_knot(elevated_surface, [0.25]; direction=1)
        refined_elevated_surface = Tesserae.NURBS.insert_knot(refined_elevated_surface, [0.5]; direction=2)
        @test nurbs_test_degrees(refined_elevated_surface) == (nurbs_test_cubic, nurbs_test_cubic)
        @test nurbs_test_knot_vectors(refined_elevated_surface)[1] == [0.0,0.0,0.0,0.0,0.25,0.5,0.5,1.0,1.0,1.0,1.0]
        @test nurbs_test_knot_vectors(refined_elevated_surface)[2] == [0.0,0.0,0.0,0.0,1/3,1/3,0.5,2/3,2/3,1.0,1.0,1.0,1.0]
        for ξ in (Vec(0.1,0.2), Vec(0.3,0.4), Vec(0.75,0.9))
            @test Tesserae.NURBS.evaluate(refined_elevated_surface, ξ) ≈ Tesserae.NURBS.evaluate(surface_control, ξ)
        end

        # Splitting a surface should produce matching boundary curves on the
        # new interface.
        left_surface, right_surface = Tesserae.NURBS.split(surface_control, 2, 0.5)
        @test size(left_surface.points) == (4, 4)
        @test size(right_surface.points) == (4, 4)
        @test nurbs_test_knot_vectors(left_surface)[2] == [0.0,0.0,0.0,1/3,0.5,0.5,0.5]
        @test nurbs_test_knot_vectors(right_surface)[2] == [0.5,0.5,0.5,2/3,1.0,1.0,1.0]
        left_interface = Tesserae.NURBS.boundaries(left_surface, 2, +1)
        right_interface = Tesserae.NURBS.boundaries(right_surface, 2, -1)
        for i in eachindex(left_interface.points)
            @test left_interface.points[i] ≈ right_interface.points[i]
        end
    end

    @testset "Modeling" begin
        # Vector sweep translates every section control point along a new
        # parametric direction.
        swept_curve = Tesserae.NURBS.sweep(curve_control, Vec(0.0,2.0); degree=nurbs_test_linear, nspans=1)
        @test size(swept_curve.points) == (4, 2)
        @test swept_curve.points[1,1] ≈ Vec(0.0, 0.0)
        @test swept_curve.points[end,end] ≈ Vec(3.0, 2.0)
        @test swept_curve.weights[:,1] == curve_weights
        @test swept_curve.weights[:,end] == curve_weights
        @test_throws ArgumentError Tesserae.NURBS.sweep(curve_control, Vec(0.0,2.0); nspans=0)

        weighted_boundary = Tesserae.NURBS.boundaries(swept_curve, 2, -1)
        @test weighted_boundary.points == curve_controlpoints
        @test weighted_boundary.weights == curve_weights

        # Trajectory sweep forms the tensor-product sum of section and
        # trajectory control nets, with rational weights multiplied.
        trajectory_control = nurbs_test_control_net(
            (nurbs_test_quadratic,),
            ([0.0,0.0,0.0,1.0,1.0,1.0],),
            [Vec(0.0,0.0), Vec(0.0,2.0), Vec(2.0,2.0)],
            [2.0, 3.0, 4.0],
        )
        swept_trajectory_curve = Tesserae.NURBS.sweep(curve_control, trajectory_control)
        @test size(swept_trajectory_curve.points) == (4, 3)
        @test swept_trajectory_curve.points[1,1] ≈ Vec(0.0, 0.0)
        @test swept_trajectory_curve.points[end,end] ≈ Vec(5.0, 2.0)
        @test swept_trajectory_curve.weights[2,3] == curve_weights[2] * trajectory_control.weights[3]

        # Lofting stacks compatible sections and keeps per-section weights.
        loft_curve = nurbs_test_control_net(
            curve_degrees,
            curve_knots,
            [Vec(Float64(i-1), 2.0) for i in 1:4],
            curve_weights,
        )
        lofted_surface = Tesserae.NURBS.loft([curve_control, loft_curve])
        @test size(lofted_surface.points) == (4, 2)
        @test lofted_surface.points[1,1] ≈ Vec(0.0, 0.0)
        @test lofted_surface.points[end,end] ≈ Vec(3.0, 2.0)
        @test lofted_surface.weights[:,1] == curve_weights
        @test lofted_surface.weights[:,2] == curve_weights

        polynomial_curve = nurbs_test_control_net(curve_degrees, curve_knots, curve_controlpoints)
        mixed_loft_surface = Tesserae.NURBS.loft([curve_control, polynomial_curve])
        @test mixed_loft_surface.weights[:,1] == curve_weights
        @test all(isone, mixed_loft_surface.weights[:,2])

        # Surface sweep and loft create volume control nets; boundary tests
        # check that the new third parametric direction is wired correctly.
        surface3d_points = map(CartesianIndices((4, 5))) do I
            Vec(Float64(I[1]-1), Float64(I[2]-1), 0.0)
        end
        surface3d_control = nurbs_test_control_net(surface_degrees, surface_knots, surface3d_points)
        solid_control = Tesserae.NURBS.sweep(surface3d_control, Vec(0.0,0.0,2.0); degree=nurbs_test_linear, nspans=2)
        @test solid_control isa Tesserae.NURBS.ControlNet
        @test size(solid_control.points) == (4, 5, 3)
        @test all(isone, solid_control.weights)
        @test solid_control.points[1,1,1] ≈ Vec(0.0, 0.0, 0.0)
        @test solid_control.points[end,end,end] ≈ Vec(3.0, 4.0, 2.0)

        solid_boundaries = Tesserae.NURBS.boundaries(solid_control)
        bottom_surface = Tesserae.NURBS.boundaries(solid_control, 3, -1)
        top_surface = Tesserae.NURBS.boundaries(solid_control, 3, +1)
        @test length(solid_boundaries) == 6
        @test bottom_surface.points == solid_control.points[:,:,1]
        @test top_surface.points == solid_control.points[:,:,end]
        @test nurbs_test_degrees(bottom_surface) == surface_degrees
        @test nurbs_test_knot_vectors(bottom_surface) == surface_knots

        offset_surface_points = map(CartesianIndices((4, 5))) do I
            Vec(Float64(I[1]-1), Float64(I[2]-1), 2.0)
        end
        offset_surface_control = nurbs_test_control_net(surface_degrees, surface_knots, offset_surface_points)
        lofted_volume = Tesserae.NURBS.loft([surface3d_control, offset_surface_control])
        @test size(lofted_volume.points) == (4, 5, 2)
        @test lofted_volume.points[1,1,1] ≈ Vec(0.0, 0.0, 0.0)
        @test lofted_volume.points[end,end,end] ≈ Vec(3.0, 4.0, 2.0)
        @test all(isone, lofted_volume.weights)

        trajectory3d_control = nurbs_test_control_net(
            (nurbs_test_linear,),
            ([0.0,0.0,1.0,1.0],),
            [Vec(0.0,0.0,0.0), Vec(0.0,0.0,2.0)],
        )
        swept_trajectory_solid = Tesserae.NURBS.sweep(surface3d_control, trajectory3d_control)
        @test size(swept_trajectory_solid.points) == (4, 5, 2)
        @test swept_trajectory_solid.points[1,1,1] ≈ Vec(0.0, 0.0, 0.0)
        @test swept_trajectory_solid.points[end,end,end] ≈ Vec(3.0, 4.0, 2.0)

        # Revolving a line around the z-axis should produce a closed rational
        # quadratic circular surface.
        line_control = nurbs_test_control_net(
            (nurbs_test_linear,),
            ([0.0,0.0,1.0,1.0],),
            [Vec(1.0,0.0,0.0), Vec(1.0,0.0,2.0)],
        )
        revolved_surface = Tesserae.NURBS.revolve(line_control, Vec(0.0,0.0,0.0), Vec(0.0,0.0,1.0))
        @test nurbs_test_degrees(revolved_surface) == (nurbs_test_linear, nurbs_test_quadratic)
        @test nurbs_test_knot_vectors(revolved_surface)[2] == [0.0,0.0,0.0,0.25,0.25,0.5,0.5,0.75,0.75,1.0,1.0,1.0]
        @test size(revolved_surface.points) == (2, 9)
        @test revolved_surface.points[1,1] ≈ Vec(1.0, 0.0, 0.0)
        @test revolved_surface.points[1,2] ≈ Vec(1.0, 1.0, 0.0)
        @test revolved_surface.points[1,3] ≈ Vec(0.0, 1.0, 0.0)
        @test revolved_surface.points[1,end] ≈ Vec(1.0, 0.0, 0.0)
        @test revolved_surface.points[2,2] ≈ Vec(1.0, 1.0, 2.0)
        @test revolved_surface.weights[1,:] ≈ [1.0, √2/2, 1.0, √2/2, 1.0, √2/2, 1.0, √2/2, 1.0]
        @test_throws ArgumentError Tesserae.NURBS.revolve(line_control, Vec(0.0,0.0,0.0), Vec(0.0,0.0,0.0))
    end
end
