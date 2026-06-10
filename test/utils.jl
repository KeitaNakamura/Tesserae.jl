@testset "Utilities" begin
    # nfill
    @test (@inferred Tesserae.nfill(3, Val(3)))   === (3,3,3)
    @test (@inferred Tesserae.nfill(1.2, Val(2))) === (1.2,1.2)
    # zero_recursive
    @test (@inferred Tesserae.zero_recursive(1)) === 0
    @test (@inferred Tesserae.zero_recursive(1.2)) === 0.0
    @test (@inferred Tesserae.zero_recursive(1.2f0)) === 0.0f0
    @test (@inferred Tesserae.zero_recursive((;x=1,y=2.0,z=3.0f0))) === (;x=0,y=0.0,z=0.0f0)
    @test (@inferred Tesserae.zero_recursive((;x=1,y=2.0,z=(;a=3.0f0)))) === (;x=0,y=0.0,z=(;a=0.0f0))
    @test (@inferred Tesserae.zero_recursive(Int)) === 0
    @test (@inferred Tesserae.zero_recursive(Float64)) === 0.0
    @test (@inferred Tesserae.zero_recursive(Float32)) === 0.0f0
    @test (@inferred Tesserae.zero_recursive(@NamedTuple{x::Int,y::Float64,z::Float32})) === (;x=0,y=0.0,z=0.0f0)
    @test (@inferred Tesserae.zero_recursive(@NamedTuple{x::Int,y::Float64,z::@NamedTuple{a::Float32}})) === (;x=0,y=0.0,z=(;a=0.0f0))
    @test_throws ArgumentError Tesserae.zero_recursive(zeros(3))
    @test_throws ArgumentError Tesserae.zero_recursive("abc")
    # fillzero!
    @test (@inferred Tesserae.fillzero!(rand(3)))::Vector{Float64} == zeros(3)
    # commas
    @test (@inferred Tesserae.commas(123))::String == "123"
    @test (@inferred Tesserae.commas(1234))::String == "1,234"
    @test (@inferred Tesserae.commas(12345678))::String == "12,345,678"
    # maparray
    @test (@inferred Tesserae.maparray(Float32, [1,2,3,4])) == Float32[1,2,3,4]
    @test (@inferred Tesserae.maparray(Float64, [1,2,3,4])) == Float64[1,2,3,4]
    @test (@inferred Tesserae.maparray(sqrt, [1,2,3,4])) == map(sqrt, [1,2,3,4])
    # Trues
    @test (@inferred Tesserae.Trues((3,2))) == trues(3,2)
    # progress
    P = Tesserae.Progress
    @test P.durationstring(60) == "0:01:00"
    @test P.timestring(0.001) == " 1.00 ms"
    @test P.speedstring(0.001) == " 1.00 ms/it"
    @test P.barstring(4, 50) == "|██  |"
    median = P.P2Median()
    @test P.median(median) == Inf
    foreach(x -> push!(median, x), (3.0, 1.0, 2.0))
    @test P.median(median) == 2.0
    push!(median, 4.0)
    @test P.median(median) == 2.5
    push!(median, NaN)
    @test median.count == 4
    for x in 5.0:100.0
        push!(median, x)
    end
    @test median.count == 100
    @test length(median.q) == 5
    # Skewed streams exercise P2 marker adjustment through the public push! path.
    high_median = P.P2Median()
    foreach(x -> push!(high_median, x), vcat(1.0:5.0, fill(1000.0, 10)))
    @test high_median.count == 15
    @test high_median.q ≈ [1.0, 89.115625, 651.938132957176, 990.8278897569446, 1000.0]
    low_median = P.P2Median()
    foreach(x -> push!(low_median, x), vcat(1.0:5.0, fill(0.0, 6)))
    @test low_median.count == 11
    @test low_median.q[1] == 0.0
    prog = P.Meter(1.0, 1.0; output=IOBuffer())
    @test P.fraction(prog, 1.0, 1.0) == 1.0
    @test P.fraction(prog, 0.0, 1.0) == 0.0
    @test P.printvalues!(prog, ()) === prog
    @test prog.numprintedvalues == 0
    io = IOBuffer()
    prog = P.Meter(0.0, 1.0; output=io, dt=0.0)
    P.update!(prog, 0.25, 1.0)
    output_text = String(take!(io))
    @test occursin("Progress:", output_text)
    @test occursin("ETA:", output_text)
    @test occursin("Elapsed:", output_text)
    @test occursin("Iterations:", output_text)
    @test occursin("Timestep:", output_text)
    @test occursin(r"Timestep: .*median", output_text)
    @test occursin("Wall time/step:", output_text)
    @test occursin(r"Wall time/step: .*median", output_text)
    showvalues = P.showvalues(prog, 0.0, 0.1)
    timestep_value = string(only(value for (name, value) in showvalues if name == :Timestep))
    step_value = string(only(value for (name, value) in showvalues if name == "Wall time/step"))
    @test first(findfirst("(median", timestep_value)) == first(findfirst("(median", step_value))
    P.finish!(prog)
    finish_output = String(take!(io))
    @test occursin("Time:", finish_output)
    @test occursin("Iterations:", finish_output)
    @test occursin("Wall time/step:", finish_output)
    @test !occursin("Histogram:", finish_output)
    color_buffer = IOBuffer()
    color_io = IOContext(color_buffer, :color => true)
    prog = P.Meter(0.0, 1.0; output=color_io, dt=0.0)
    P.update!(prog, 0.25, 1.0)
    take!(color_buffer)
    P.finish!(prog)
    @test occursin("\e[34mWall time/step:", String(take!(color_buffer)))
    prog = P.Meter(0.0, 10.0; output=IOBuffer())
    for i in 1:5
        P.record_step!(prog, i / 1000)
    end
    @test prog.step_count == 5
    @test prog.step_min == 0.001
    @test prog.step_max == 0.005
    @test prog.step_mean ≈ 0.003
    @test P.median(prog.step_median) ≈ 0.003
    @test P.step_std(prog) ≈ sqrt(2.5e-6)
    prog = P.Meter(0.0, 10.0; output=IOBuffer())
    @test P.eta(prog) == "N/A"
    P.update_eta!(prog, 1.0, 1.0)
    @test prog.eta ≈ 9.0
    P.update_eta!(prog, 2.0, 2.0)
    @test prog.eta ≈ 12.0
    io = IOBuffer()
    prog = P.Meter(0.0, 1.0; output=io, dt=0.0, summary=false)
    P.update!(prog, 0.5, 1.0)
    P.update!(prog, 1.0, 1.0)
    @test prog.done
    @test occursin("100%", String(take!(io)))
end
