using FileIO, MeshIO

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
