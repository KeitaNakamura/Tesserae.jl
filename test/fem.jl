@testset "FEM field and geometry" begin
    @testset "Quadrilateral domain" begin
        α = 0.2
        geometry = FEMesh(
            Tesserae.Quad9(),
            [
                Vec(0.0, 0.0), Vec(1.0, 0.0), Vec(1.0, 1.0), Vec(0.0, 1.0),
                Vec(0.5, 0.0), Vec(1.0, 0.5), Vec(0.5, 1 + α), Vec(0.0, 0.5), Vec(0.5, (1 + α) / 2),
            ],
            [Tesserae.SVector(1, 2, 3, 4, 5, 6, 7, 8, 9)],
        )
        field = FEMesh(
            Tesserae.Quad4(),
            [Vec(-1.0, -1.0), geometry.nodes[1:4]...],
            [Tesserae.SVector(2, 3, 4, 5)],
        )
        rule = generate_quadrature_rule(Tesserae.Quad9())
        points = @inferred generate_particles(@NamedTuple{x::Vec{2,Float64}, V::Float64}, geometry, rule)
        weights = @inferred generate_basis_weights(field, size(points); name=Val(:N))

        @test_throws ArgumentError generate_particles(@NamedTuple{x::Vec{2,Float64}}, geometry, generate_quadrature_rule(Tesserae.Tri6()))
        @test quadrature_rule(points) === rule
        @test points[1,1] == parent(points)[1,1]
        points_view = @inferred view(points, :, [1])
        @test quadrature_rule(points_view) === rule
        original_point = parent(points)[1,1]
        parent(points)[1,1] = merge(original_point, (; V=1))
        @test points_view[1,1].V == 1
        points_view[1,1] = original_point
        @test parent(points)[1,1] == original_point
        @test collect(points_view) == collect(parent(points_view))
        adapted_points = Tesserae.Adapt.adapt(Array, points)
        @test adapted_points isa QuadraturePoints
        @test parent(adapted_points) isa Tesserae.StructArray
        @test quadrature_rule(adapted_points) === rule
        @test supportnodes(weights[1,1]) == Tesserae.SVector(2, 3, 4, 5)
        @test (@inferred update!(weights, points, geometry; measure=points.V)) === weights
        for (q, (point, weight)) in enumerate(zip(rule.points, rule.weights))
            ξ, η = point
            N, dNdξ = Tesserae.jet(Order(1), Tesserae.Quad4(), point)
            J = Mat{2,2}(0.5, -α * ξ * (1 + η), 0, (1 + α * (1 - ξ^2)) / 2)
            @test points.x[q,1] ≈ Vec((1 + ξ) / 2, (1 + η) * (1 + α * (1 - ξ^2)) / 2)
            @test weights[q,1].N ≈ N
            @test weights[q,1].∇N ≈ dNdξ .⊡ Ref(inv(J))
            @test supportnodes(weights[q,1]) == Tesserae.SVector(2, 3, 4, 5)
            @test points.V[q,1] ≈ weight * det(J)
        end
        @test sum(points.V) ≈ 1 + 2α / 3
    end

    @testset "Higher-order field" begin
        geometry = FEMesh(Tesserae.Quad4(), [Vec(0.0, 0.0), Vec(1.0, 0.0), Vec(1.0, 1.0), Vec(0.0, 1.0)], [Tesserae.SVector(1, 2, 3, 4)])
        field = FEMesh(
            Tesserae.Quad9(),
            [Vec(-1.0, -1.0), Vec(0.0, 0.0), Vec(1.0, 0.0), Vec(1.0, 1.0), Vec(0.0, 1.0), Vec(0.5, 0.0), Vec(1.0, 0.5), Vec(0.5, 1.0), Vec(0.0, 0.5), Vec(0.5, 0.5)],
            [Tesserae.SVector(2, 3, 4, 5, 6, 7, 8, 9, 10)],
        )
        rule = generate_quadrature_rule(Tesserae.Quad9())
        points = generate_particles(@NamedTuple{x::Vec{2,Float64}, V::Float64}, geometry, rule)
        weights = generate_basis_weights(field, size(points); name=Val(:N))
        J = Mat{2,2}(0.5, 0, 0, 0.5)

        @test supportnodes(weights[1,1]) == Tesserae.SVector(2, 3, 4, 5, 6, 7, 8, 9, 10)
        value_weights = generate_basis_weights(field, size(points); derivative=Order(0))
        @test_throws ArgumentError update!(value_weights, points, geometry)
        @test update!(weights, points, geometry; measure=points.V) === weights
        @test all(q -> weights[q,1].N ≈ Tesserae.value(Tesserae.Quad9(), rule.points[q]), eachindex(rule.points))
        @test all(q -> weights[q,1].∇N ≈ last(Tesserae.jet(Order(1), Tesserae.Quad9(), rule.points[q])) .⊡ Ref(inv(J)), eachindex(rule.points))
        @test points.V[:,1] ≈ rule.weights / 4
    end

    @testset "Triangular domain" begin
        α = 0.3
        geometry = FEMesh(
            Tesserae.Tri6(),
            [Vec(0.0, 0.0), Vec(1.0, 0.0), Vec(0.0, 1.0), Vec(0.5, 0.0), Vec(0.0, 0.5), Vec(0.5, 0.5 + α / 4)],
            [Tesserae.SVector(1, 2, 3, 4, 5, 6)],
        )
        field = only(generate_field_meshes((geometry,), Order(1)))
        rule = generate_quadrature_rule(Tesserae.Tri6())
        points = generate_particles(@NamedTuple{x::Vec{2,Float64}, V::Float64}, geometry, rule)
        weights = generate_basis_weights(field, size(points); name=Val(:N))

        update!(weights, points, geometry; measure=points.V)
        for (q, (point, weight)) in enumerate(zip(rule.points, rule.weights))
            @test weights[q,1].N ≈ Tesserae.value(Tesserae.Tri3(), point)
            @test supportnodes(weights[q,1]) == Tesserae.SVector(1, 2, 3)
            @test points.V[q,1] ≈ weight * (1 + α * point[1])
        end
        @test sum(points.V) ≈ 1/2 + α/6
    end

    @testset "Curved boundary" begin
        h = 0.25
        geometry = FEMesh(Tesserae.Line3(), [Vec(0.0, 0.0), Vec(1.0, 0.0), Vec(0.5, h)], [Tesserae.SVector(1, 2, 3)])
        field = FEMesh(Tesserae.Line2(), [Vec(-1.0, -1.0), geometry.nodes[1:2]...], [Tesserae.SVector(2, 3)])
        rule = generate_quadrature_rule(Tesserae.Line3())
        points = generate_particles(@NamedTuple{x::Vec{2,Float64}, dS::Float64, n::Vec{2,Float64}}, geometry, rule)
        weights = generate_basis_weights(field, size(points); name=Val(:N))

        update!(weights, points, geometry; measure=points.dS, normal=points.n)
        for (q, (point, weight)) in enumerate(zip(rule.points, rule.weights))
            tangent = Vec(0.5, -2h * point[1])
            scale = norm(tangent)
            @test weights[q,1].N ≈ Tesserae.value(Tesserae.Line2(), point)
            @test supportnodes(weights[q,1]) == Tesserae.SVector(2, 3)
            @test points.dS[q,1] ≈ weight * scale
            @test points.n[q,1] ≈ Vec(tangent[2], -tangent[1]) / scale
        end
    end

    @testset "Empty domain" begin
        mesh = FEMesh(Tesserae.Quad4(), Vec{2,Float64}[], Tesserae.SVector{4,Int}[])
        points = generate_particles(@NamedTuple{x::Vec{2,Float64}, V::Float64}, mesh)
        weights = generate_basis_weights(mesh, size(points))
        @test update!(weights, points, mesh; measure=points.V) === weights
    end
end
