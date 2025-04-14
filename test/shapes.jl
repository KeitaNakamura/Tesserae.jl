@testset "Shape" begin
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
end
