@testset "Shape" begin
    shapes = (
        Tesserae.Line2(), Tesserae.Line3(), Tesserae.Line4(),
        Tesserae.Quad4(), Tesserae.Quad8(), Tesserae.Quad9(),
        Tesserae.Tri3(), Tesserae.Tri6(),
        Tesserae.Hex8(), Tesserae.Hex20(), Tesserae.Hex27(),
        Tesserae.Tet4(), Tesserae.Tet10(),
    )

    # check primary nodes
    @test Tesserae.primarynodes_indices(Tesserae.Line2()) ==
          Tesserae.primarynodes_indices(Tesserae.Line3()) ==
          Tesserae.primarynodes_indices(Tesserae.Line4())
    @test Tesserae.primarynodes_indices(Tesserae.Quad4()) ==
          Tesserae.primarynodes_indices(Tesserae.Quad8()) ==
          Tesserae.primarynodes_indices(Tesserae.Quad9())
    @test Tesserae.primarynodes_indices(Tesserae.Hex8()) ==
          Tesserae.primarynodes_indices(Tesserae.Hex20()) ==
          Tesserae.primarynodes_indices(Tesserae.Hex27())
    @test Tesserae.primarynodes_indices(Tesserae.Tri3()) ==
          Tesserae.primarynodes_indices(Tesserae.Tri6())
    @test Tesserae.primarynodes_indices(Tesserae.Tet4()) ==
          Tesserae.primarynodes_indices(Tesserae.Tet10())

    @testset "shape function invariants" begin
        reference_measure(shape) =
            shape isa Tesserae.Line ? 2 :
            shape isa Tesserae.Quad ? 4 :
            shape isa Tesserae.Tri  ? 1/2 :
            shape isa Tesserae.Hex  ? 8 :
            shape isa Tesserae.Tet  ? 1/6 : error()

        for shape in shapes
            nodes = Tesserae.localnodes(shape)
            @test length(nodes) == Tesserae.nlocalnodes(shape)
            for i in eachindex(nodes)
                values = Tesserae.value(shape, nodes[i])
                @test values[i] ≈ 1
                @test sum(values) ≈ 1
                @test all(j -> j == i || values[j] ≈ 0, eachindex(values))
            end
            rule = generate_quadrature_rule(shape)
            @test sum(rule.weights) ≈ reference_measure(shape)
        end
    end

    @testset "quadrature exactness" begin
        integrate(rule, powers) = sum(w * prod(xᵢ^p for (xᵢ, p) in zip(x, powers)) for (x, w) in zip(rule.points, rule.weights))

        for T in (Float16, Float32, Float64)
            typed_rule = generate_quadrature_rule(T, Tesserae.Quad, 4)
            @test eltype(first(typed_rule.points)) === T
            @test eltype(typed_rule.weights) === T
        end
        @test eltype(generate_quadrature_rule(Float32, Tesserae.Quad9()).weights) === Float32
        @test eltype(generate_quadrature_rule(Tesserae.Quad, 4).weights) === Float64
        @test_throws ArgumentError generate_quadrature_rule(BigFloat, Tesserae.Quad, 4)

        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Line, 11), (10,)), 2/11; atol=1e-14)
        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Line, 12), (12,)), 2/13; atol=1e-14)
        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Quad, 4), (4, 4)), (2/5)^2; atol=1e-14)
        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Hex, 4), (4, 4, 4)), (2/5)^3; atol=1e-14)
        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Tri, 4), (2, 2)), 1/180; atol=1e-14)
        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Tri, 6), (3, 3)), 1/1120; atol=1e-14)
        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Tet, 5), (3, 1, 1)), 1/6720; atol=1e-14)
        @test isapprox(integrate(generate_quadrature_rule(Tesserae.Tet, 7), (3, 2, 2)), 1/151200; atol=1e-14)

        @test_throws ArgumentError generate_quadrature_rule(Tesserae.Line, -1)
        @test_throws ArgumentError generate_quadrature_rule(Tesserae.Tri, 7)
        @test_throws ArgumentError generate_quadrature_rule(Tesserae.Tet, 8)
    end

    @testset "shape quadrature" begin
        standard_rules = (
            (Tesserae.Line2(), 2), (Tesserae.Line3(), 3), (Tesserae.Line4(), 4),
            (Tesserae.Quad4(), 4), (Tesserae.Quad8(), 9), (Tesserae.Quad9(), 9),
            (Tesserae.Hex8(), 8), (Tesserae.Hex20(), 27), (Tesserae.Hex27(), 27),
            (Tesserae.Tri3(), 3), (Tesserae.Tri6(), 6),
            (Tesserae.Tet4(), 4), (Tesserae.Tet10(), 14),
        )

        for (shape, npoints) in standard_rules
            rule = generate_quadrature_rule(shape)
            @test length(rule.points) == npoints

            n = Tesserae.nlocalnodes(shape)
            mass = zeros(n, n)
            for (point, weight) in zip(rule.points, rule.weights)
                values = Tesserae.value(shape, point)
                mass .+= weight .* (values * values')
            end
            @test rank(mass) == n
        end
    end
end
