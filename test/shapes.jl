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
            @test sum(Tesserae.quadweights(shape)) ≈ reference_measure(shape)
        end
    end
end
