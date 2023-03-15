using Marble
using Random
using Test

using ReadVTK
using NaturalSort

include("sparray.jl")
include("lattice.jl")
include("particles.jl")
include("interpolations.jl")
include("blockspace.jl")
include("transfer.jl")

const FIX_RESULTS = false

function check_example(testname::String, case, interp, transfer=DefaultTransfer(); dx, kwargs...)
    @testset "$testname" begin
        outdir = joinpath("examples", "$testname.tmp")
        rm(outdir; recursive = true, force = true)

        include(joinpath("../examples", "$testname.jl"))
        @eval $(Symbol(testname))($interp, $transfer; outdir=$outdir, $kwargs...)

        result_file = joinpath(
            outdir,
            sort(filter(file -> endswith(file, ".vtu"), only(walkdir(outdir))[3]), lt=natural)[end],
        )

        if FIX_RESULTS
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
    check_example("SandColumn", 1, QuadraticBSpline(), FLIP(); dx)
    check_example("SandColumn", 2, LinearWLS(QuadraticBSpline()); dx)
    check_example("SandColumn", 2, LinearWLS(QuadraticBSpline()), TPIC(); dx)
    check_example("SandColumn", 3, LinearWLS(QuadraticBSpline()), APIC(); dx)
    check_example("SandColumn", 4, KernelCorrection(QuadraticBSpline()), TPIC(); dx)
    check_example("SandColumn", 5, KernelCorrection(QuadraticBSpline()), APIC(); dx)
    check_example("SandColumn", 6, KernelCorrection(QuadraticBSpline()), AFLIP(); dx)
    # StripFooting
    dx = 0.125
    lockingfree = true
    check_example("StripFooting", 1, LinearBSpline(); dx, lockingfree)
    check_example("StripFooting", 2, uGIMP(); dx, lockingfree, CFL=0.5)
    check_example("StripFooting", 3, LinearWLS(QuadraticBSpline()); dx, lockingfree)
    check_example("StripFooting", 4, KernelCorrection(QuadraticBSpline()), TPIC(); dx, lockingfree)
    check_example("StripFooting", 5, KernelCorrection(QuadraticBSpline()), APIC(); dx, lockingfree)
    # DamBreak
    dx = 0.07
    t_stop = 1.0
    check_example("DamBreak", 1, QuadraticBSpline(); dx, t_stop)
    check_example("DamBreak", 2, uGIMP(); dx, t_stop)
    check_example("DamBreak", 3, LinearWLS(QuadraticBSpline()); dx, t_stop)
    check_example("DamBreak", 4, KernelCorrection(QuadraticBSpline()), TPIC(); dx, t_stop)
    check_example("DamBreak", 5, KernelCorrection(QuadraticBSpline()), APIC(); dx, t_stop)
end
