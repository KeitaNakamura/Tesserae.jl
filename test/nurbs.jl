nurbs_test_axes(degrees, knot_vectors) = map(Tesserae.NURBS.BSplineAxis, degrees, knot_vectors)
nurbs_test_control_net(degrees, knot_vectors, points, weights...) = Tesserae.NURBS.ControlNet(nurbs_test_axes(degrees, knot_vectors), points, weights...)
nurbs_test_degrees(net::Tesserae.NURBS.ControlNet) = map(axis -> axis.degree, net.axes)
nurbs_test_knot_vectors(net::Tesserae.NURBS.ControlNet) = map(axis -> axis.knot_vector, net.axes)

const nurbs_test_linear = Tesserae.NURBS.linear
const nurbs_test_quadratic = Tesserae.NURBS.quadratic
const nurbs_test_cubic = Tesserae.NURBS.cubic

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
