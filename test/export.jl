using FileIO, MeshIO

@testset "PLY" begin
    xs = [rand(Vec{3,Float64}) for i in 1:100]
    cd(tempdir()) do
        Tesserae.write_ply("plyfile", xs)
        mesh = load("plyfile.ply")
        @test reinterpret(Float32, map(Vec{3,Float32}, xs)) ≈ reinterpret(Float32, mesh.position)
    end
    xs = [rand(Vec{2,Float64}) for i in 1:100]
    cd(tempdir()) do
        Tesserae.write_ply("plyfile", xs)
        mesh = load("plyfile.ply")
        @test reinterpret(Float32, map(x -> Vec{3,Float32}([x;0]), xs)) ≈ reinterpret(Float32, mesh.position)
    end
end
