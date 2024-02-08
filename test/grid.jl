@testset "Grid" begin
    Δx = 0.2
    xlims = (0,3)
    ylims = (0,4)
    lattice = Lattice(Δx, xlims, ylims)

    # constructors
    ## without grid property
    @test (@inferred generate_grid(lattice)).x === lattice
    @test (@inferred generate_grid(spacing(lattice), xlims, ylims)).x == lattice
    @test (@inferred spacing(generate_grid(lattice))) === spacing(lattice)
    @test (@inferred spacing_inv(generate_grid(lattice))) === spacing_inv(lattice)
    ## with grid property (Array)
    for grid in ((@inferred generate_grid(@NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, Δx, xlims, ylims)),
                 (@inferred generate_grid(Array, @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, Δx, xlims, ylims)),)
        @test grid isa Grid
        @test (grid.x)::Lattice{2, Float64} == lattice
        @test all(iszero, (grid.m)::Array{Float64, 2})
        @test all(iszero, (grid.v)::Array{Vec{2,Float64}, 2})
        grid.m .= rand()
        grid.v .= rand(Vec{2})
        fillzero!(grid)
        @test grid.x == lattice
        @test all(iszero, grid.m)
        @test all(iszero, grid.v)
    end
    @test_throws Exception generate_grid(@NamedTuple{x::Vec{3,Float64}, m::Float64, v::Vec{3,Float64}}, Δx, xlims, ylims)
    @test_throws Exception generate_grid(Array, @NamedTuple{x::Vec{3,Float64}, m::Float64, v::Vec{3,Float64}}, Δx, xlims, ylims)
    @test_throws Exception generate_grid(Array, @NamedTuple{x::Vec{2,Float64}, v::Vector{Float64}}, Δx, xlims, ylims)
    ## with grid property (SpArray)
    grid = (@inferred generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, Δx, xlims, ylims))
    @test grid isa Grid
    @test grid isa SpGrid
    @test (grid.x)::Lattice{2, Float64} == lattice
    @test all(iszero, (grid.m)::SpArray{Float64, 2})
    @test all(iszero, (grid.v)::SpArray{Vec{2,Float64}, 2})
    @test !all(i->isactive(grid,i), eachindex(grid))
    @test all(x->Sequoia.get_spinds(x)===Sequoia.get_spinds(grid), (grid.m, grid.v))
    @test_throws Exception generate_grid(SpArray, @NamedTuple{x::Vec{3,Float64}, m::Float64, v::Vec{3,Float64}}, Δx, xlims, ylims)
    @test_throws Exception generate_grid(SpArray, @NamedTuple{x::Vec{2,Float64}, v::Vector{Float64}}, Δx, xlims, ylims)

    # first entry becomes lattice
    grid = generate_grid(@NamedTuple{v::Vec{3,Float32}, m::Float32, x::Vec{3,Float32}}, Δx, xlims, ylims, ylims)
    @test grid.v isa Lattice{3,Float32}
    @test all(iszero, (grid.m)::Array{Float32,3})
    @test all(iszero, (grid.x)::Array{Vec{3,Float32},3})
end
