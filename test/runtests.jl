using Poingr
using Random
using Test

using ReadVTK
using NaturalSort

using Poingr: Index

struct NodeState
    a::Float64
    b::Float64
end

include("utils.jl")
include("sparray.jl")
include("grid.jl")
include("shapefunctions.jl")

include("mpcache.jl")

include("MaterialModels/SoilHyperelastic.jl")
include("MaterialModels/DruckerPrager.jl")

const fix_results = false

function check_example(testname::String, case, shape_function; kwargs...)
    @testset "$testname" begin
        include(joinpath("../examples", "$testname.jl"))
        @eval $(Symbol(testname))($shape_function; show_progress = false, $kwargs...)

        output_dir = joinpath("../examples", "$testname.tmp")
        result_file = joinpath(
            output_dir,
            sort(filter(file -> endswith(file, ".vtu"), only(walkdir(output_dir))[3]), lt = natural)[end],
        )

        if fix_results
            mv(result_file, joinpath("examples", "$testname$case.vtu"); force = true)
        else
            # check results
            expected = VTKFile(joinpath("examples", "$testname$case.vtu")) # expected output
            result = VTKFile(result_file)
            expected_points = get_points(expected)
            result_points = get_points(result)
            @assert size(expected_points) == size(result_points)
            @test all(eachindex(expected_points)) do i
                val = expected_points[i]
                0.99*val ≤ result_points[i] ≤ 1.01*val # ±1%
            end
        end
    end
end

@testset "Check examples" begin
    check_example("sandcolumn", 1, QuadraticBSpline())
    check_example("sandcolumn", 2, LinearWLS(QuadraticBSpline()))
    check_example("sandcolumn", 2, LinearWLS(QuadraticBSpline()); affine_transfer = true)
    check_example("sandcolumn", 3, BilinearWLS(QuadraticBSpline()))
    check_example("stripfooting", 1, LinearBSpline())
    check_example("stripfooting", 2, GIMP())
    check_example("stripfooting", 3, LinearWLS(QuadraticBSpline()))
end
