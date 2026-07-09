using FileIO, MeshIO

@testset "VTK helpers" begin
    @test Tesserae.to_vtk_celltype(Tesserae.Line2()) === Tesserae.WriteVTK.VTKCellTypes.VTK_LINE
    @test Tesserae.to_vtk_celltype(Tesserae.Tri6()) === Tesserae.WriteVTK.VTKCellTypes.VTK_QUADRATIC_TRIANGLE
    @test Tesserae.to_vtk_celltype(Tesserae.Hex27()) === Tesserae.WriteVTK.VTKCellTypes.VTK_TRIQUADRATIC_HEXAHEDRON

    @test Tuple(Tesserae.to_vtk_connectivity(Tesserae.Tri6())) == (1, 2, 3, 4, 6, 5)
    @test Tuple(Tesserae.to_vtk_connectivity(Tesserae.Tet10())) == (1, 2, 3, 4, 5, 8, 6, 7, 10, 9)
    @test Tuple(Tesserae.to_vtk_connectivity(Tesserae.Hex20())) ==
          (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 10, 17, 19, 20, 18, 11, 13, 15, 16)

    tensor_data = [symmetric(Vec(1.0, 2.0) ⊗ Vec(3.0, 4.0))]
    formatted = Tesserae.vtk_format(tensor_data)
    @test size(formatted) == (6, 1)
    @test isequal(formatted[:, 1], [3.0, 8.0, NaN, 5.0, NaN, NaN])

    mktempdir() do dir
        cd(dir) do
            files = openvtk("quad4", FEMesh(Tesserae.Quad4(), CartesianMesh(1, (0,1), (0,1)))) do vtk
                vtk["temperature"] = [1.0, 2.0, 3.0, 4.0]
            end
            @test length(files) == 1
            @test isfile(only(files))
            @test filesize(only(files)) > 0
        end
    end
end

@testset "PLY" begin
    xs = [
        Vec(0.1, 0.2, 0.3),
        Vec(1.0, -2.0, 3.5),
        Vec(-0.4, 0.0, 2.0),
        Vec(2.5, 1.5, -1.0),
    ]
    cd(tempdir()) do
        Tesserae.write_ply("plyfile", xs)
        mesh = load("plyfile.ply")
        @test reinterpret(Float32, map(Vec{3,Float32}, xs)) ≈ reinterpret(Float32, mesh.position)
    end
    xs = [
        Vec(0.1, 0.2),
        Vec(1.0, -2.0),
        Vec(-0.4, 0.0),
        Vec(2.5, 1.5),
    ]
    cd(tempdir()) do
        Tesserae.write_ply("plyfile", xs)
        mesh = load("plyfile.ply")
        @test reinterpret(Float32, map(x -> Vec{3,Float32}([x;0]), xs)) ≈ reinterpret(Float32, mesh.position)
    end
end
