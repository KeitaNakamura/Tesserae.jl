@testset "Grid" begin
    h = 0.2
    xlims = (0,3)
    ylims = (0,4)
    mesh = CartesianMesh(h, xlims, ylims)

    # constructors
    ## with Array
    for grid in ((@inferred generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, mesh)),
                 (@inferred generate_grid(Array, @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, mesh)),)
        @test grid isa Grid
        @test (grid.x)::CartesianMesh{2, Float64} == mesh
        @test all(iszero, (grid.m)::Array{Float64, 2})
        @test all(iszero, (grid.v)::Array{Vec{2,Float64}, 2})
        grid.m .= rand()
        grid.v .= Ref(rand(Vec{2}))
        Tesserae.fillzero!(grid)
        @test grid.x == mesh
        @test all(iszero, grid.m)
        @test all(iszero, grid.v)
    end
    # wrong type of `x`
    @test_throws Exception generate_grid(@NamedTuple{x::Vec{3,Float64}, m::Float64, v::Vec{3,Float64}}, mesh)
    @test_throws Exception generate_grid(Array, @NamedTuple{x::Vec{3,Float64}, m::Float64, v::Vec{3,Float64}}, mesh)
    @test_throws Exception generate_grid(Array, @NamedTuple{x::Vec{2,Float32}, m::Float64, v::Vec{3,Float64}}, mesh)
    # given type must be `isbitstype`
    @test_throws Exception generate_grid(Array, @NamedTuple{x::Vec{2,Float64}, v::Vector{Float64}}, mesh)

    ## with SpArray
    grid = (@inferred generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, mesh))
    @test grid isa Grid
    @test grid isa SpGrid
    @test (grid.x)::CartesianMesh{2, Float64} == mesh
    @test all(iszero, (grid.m)::SpArray{Float64, 2})
    @test all(iszero, (grid.v)::SpArray{Vec{2,Float64}, 2})
    @test !all(i->Tesserae.isactive(grid,i), eachindex(grid))
    @test all(x->Tesserae.get_spinds(x)===Tesserae.get_spinds(grid), (grid.m, grid.v))
    mesh_block3 = CartesianMesh(h, xlims, ylims; block_size_log2=Val(3))
    grid_block3 = generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, m::Float64}, mesh_block3)
    @test Tesserae.block_size_log2(Tesserae.get_spinds(grid_block3)) === 3
    @test Tesserae.nblocks(Tesserae.get_spinds(grid_block3)) === Tesserae.nblocks(mesh_block3)
    # wrong type of `x`
    @test_throws Exception generate_grid(SpArray, @NamedTuple{x::Vec{3,Float64}, m::Float64, v::Vec{3,Float64}}, mesh)
    # give type must be `isbitstype`
    @test_throws Exception generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, v::Vector{Float64}}, mesh)

    # first entry becomes mesh
    grid = generate_grid(@NamedTuple{v::Vec{3,Float32}, m::Float32, x::Vec{3,Float32}}, CartesianMesh(Float32, h, xlims, ylims, ylims))
    @test grid.v isa CartesianMesh{3,Float32}
    @test all(iszero, (grid.m)::Array{Float32,3})
    @test all(iszero, (grid.x)::Array{Vec{3,Float32},3})

    # broadcast for SpArray
    grid = generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, mesh)
    spinds = Tesserae.get_spinds(grid)
    blkspy = rand(Bool, Tesserae.nblocks(spinds))
    update_sparsity!(grid, blkspy)
    block_size = Val(Tesserae.block_size_log2(spinds))
    active_indices = filter(eachindex(grid)) do I
        block, _ = Tesserae.global_to_blocklocal(Tuple(I)...; block_size_log2=block_size)
        blkspy[block...]
    end
    @test all(i->Tesserae.isactive(grid,i), active_indices)
    grid.m .= rand(size(grid))
    grid.v .= grid.x .* rand(size(grid))
    array_x = Array(grid.x)
    array_m = Array(grid.m)
    array_v = Array(grid.v)
    # broadcast `SpArray`s having identical `SpIndices`
    @. grid.v = grid.v - grid.v / grid.m
    @test grid.v == (@. array_v = array_v - array_v / array_m * !iszero(array_m))
    # broadcast `SpArray`s and `AbstractArray`
    @. grid.v = grid.v - grid.x / grid.m
    @test grid.v == (@. array_v = array_v - array_x / array_m * !iszero(array_m))
end
