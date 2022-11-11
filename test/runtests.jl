using Marble
using Random
using Test

using ReadVTK
using NaturalSort

include("misc.jl")
include("sparray.jl")
include("grid.jl")
include("interpolations.jl")

include("pointstate.jl")
include("mpspace.jl")
include("transfer.jl")

const fix_results = false

function check_example(testname::String, case, interp, transfer=Transfer(interp); dx, kwargs...)
    @testset "$testname" begin
        outdir = joinpath("examples", "$testname.tmp")
        rm(outdir; recursive = true, force = true)

        include(joinpath("../examples", "$testname.jl"))
        @eval $(Symbol(testname))($interp, $transfer; showprogress=false, outdir=$outdir, $kwargs...)

        result_file = joinpath(
            outdir,
            sort(filter(file -> endswith(file, ".vtu"), only(walkdir(outdir))[3]), lt=natural)[end],
        )

        if fix_results
            mv(result_file, joinpath("examples", "$testname$case.vtu"); force=true)
        else
            # check results
            expected = VTKFile(joinpath("examples", "$testname$case.vtu")) # expected output
            result = VTKFile(result_file)
            expected_points = get_points(expected)
            result_points = get_points(result)
            @assert size(expected_points) == size(result_points)
            @test all(eachindex(expected_points)) do i
                norm(expected_points[i] - result_points[i]) < 0.1dx
            end
        end
    end
end

@testset "Check examples" begin
    # SandColumn
    dx = 0.01
    check_example("SandColumn", 1, QuadraticBSpline(); dx)
    check_example("SandColumn", 2, LinearWLS(QuadraticBSpline()); dx)
    check_example("SandColumn", 2, LinearWLS(QuadraticBSpline()), TPIC(); dx)
    check_example("SandColumn", 3, LinearWLS(QuadraticBSpline()), APIC(); dx)
    check_example("SandColumn", 4, KernelCorrection(QuadraticBSpline()), TPIC(); dx)
    check_example("SandColumn", 5, KernelCorrection(QuadraticBSpline()), APIC(); dx)
    check_example("SandColumn", 6, KernelCorrection(QuadraticBSpline()), AFLIP(); dx)
    # StripFooting
    dx = 0.125
    ν = 0.3
    handle_volumetric_locking = true
    check_example("StripFooting", 1, LinearBSpline(); dx, ν, handle_volumetric_locking)
    check_example("StripFooting", 2, GIMP(); dx, ν, handle_volumetric_locking, CFL=0.5)
    check_example("StripFooting", 3, LinearWLS(QuadraticBSpline()); dx, ν, handle_volumetric_locking)
    check_example("StripFooting", 4, KernelCorrection(QuadraticBSpline()), TPIC(); dx, ν, handle_volumetric_locking)
    check_example("StripFooting", 5, KernelCorrection(QuadraticBSpline()), APIC(); dx, ν, handle_volumetric_locking)
end
