@testset "MPValue" begin
    for dim in (1,2,3)
        @test eltype(MPValue(Vec{dim}, QuadraticBSpline())) ==
              eltype(MPValue(Vec{dim, Float64}, QuadraticBSpline()))
        for T in (Float32, Float64)
            for kernel in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(), GIMP())
                for extension in (identity, KernelCorrection)
                    it = extension(kernel)
                    mp = @inferred MPValue(Vec{dim,T}, it)
                    @test Sequoia.interpolation(mp) === it
                    @test mp.w isa Array{T}
                    @test mp.∇w isa Array{Vec{dim,T}}
                    @test ndims(mp.w) == dim
                    @test ndims(mp.∇w) == dim
                    @test size(mp.w) == size(mp.∇w)
                    @test all(size(neighboringnodes(mp)) .< size(mp.w))

                    # diff = nothing
                    mp = @inferred MPValue(Vec{dim,T}, it; diff=nothing)
                    @test hasproperty(mp, :w) && mp.w isa Array{T}
                    @test !hasproperty(mp, :∇w)
                    @test !hasproperty(mp, :∇∇w)
                    @test ndims(mp.w) == dim
                    # diff = gradient
                    mp = @inferred MPValue(Vec{dim,T}, it; diff=gradient)
                    @test hasproperty(mp, :w)  && mp.w  isa Array{T}
                    @test hasproperty(mp, :∇w) && mp.∇w isa Array{Vec{dim,T}}
                    @test !hasproperty(mp, :∇∇w)
                    @test ndims(mp.w) == ndims(mp.∇w) == dim
                    @test size(mp.w) == size(mp.∇w)
                    # diff = hessian
                    mp = @inferred MPValue(Vec{dim,T}, it; diff=hessian)
                    @test hasproperty(mp, :w)   && mp.w   isa Array{T}
                    @test hasproperty(mp, :∇w)  && mp.∇w  isa Array{Vec{dim,T}}
                    @test hasproperty(mp, :∇∇w) && mp.∇∇w isa Array{<: SymmetricSecondOrderTensor{dim,T}}
                    @test ndims(mp.w) == ndims(mp.∇w) == ndims(mp.∇∇w) == dim
                    @test size(mp.w) == size(mp.∇w) == size(mp.∇∇w)
                end
            end
        end
    end
end

@testset "MPValueVector" begin
    for dim in (1,2,3)
        @test eltype(generate_mpvalues(Vec{dim}, QuadraticBSpline(), 2)) ==
              eltype(generate_mpvalues(Vec{dim, Float64}, QuadraticBSpline(), 2))
        for T in (Float32, Float64)
            n = 100
            mpvalues = @inferred generate_mpvalues(Vec{dim,T}, QuadraticBSpline(), n)
            @test size(mpvalues) === (n,)
            @test Sequoia.interpolation(mpvalues) === QuadraticBSpline()
            @test all(eachindex(mpvalues)) do i
                typeof(mpvalues[1]) === eltype(mpvalues)
            end
        end
    end
end

@testset "Interpolations" begin

    function check_partition_of_unity(mp, x; atol=sqrt(eps(eltype(mp.w))))
        indices = neighboringnodes(mp)
        CI = CartesianIndices(indices) # local indices
        isapprox(sum(mp.w[CI]), 1) && isapprox(sum(mp.∇w[CI]), zero(eltype(mp.∇w)); atol)
    end
    function check_linear_field_reproduction(mp, x, X; atol=sqrt(eps(eltype(mp.w))))
        indices = neighboringnodes(mp)
        CI = CartesianIndices(indices) # local indices
        isapprox(mapreduce((j,i) -> X[i]*mp.w[j],  +, CI, indices), x; atol) &&
        isapprox(mapreduce((j,i) -> X[i]⊗mp.∇w[j], +, CI, indices), I; atol)
    end

    @testset "$it" for it in (LinearBSpline(), QuadraticBSpline(), CubicBSpline())
        for T in (Float32, Float64), dim in (1,2,3)
            Random.seed!(1234)
            mp = MPValue(Vec{dim,T}, it)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            @test all(1:100) do _
                x = rand(Vec{dim, T})
                update!(mp, x, mesh)
                isnearbounds = size(mp.w) != size(neighboringnodes(mp))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                isnearbounds ? (PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "GIMP()" begin
        it = GIMP()
        for T in (Float32, Float64), dim in (1,2,3)
            Random.seed!(1234)
            mp = MPValue(Vec{dim,T}, it)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            l = 0.5*spacing(mesh) / 2
            @test all(1:100) do _
                x = rand(Vec{dim, T})
                update!(mp, (;x,l), mesh)
                isnearbounds = any(.!(l .< x .< 1-l))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                # GIMP doesn't have pertition of unity when very closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                isnearbounds ? (!PU && !LFR) : (PU && LFR) # GIMP
            end
        end
    end

    @testset "KernelCorrection($kernel)" for kernel in (LinearBSpline(), QuadraticBSpline(), CubicBSpline(), GIMP())
        it = KernelCorrection(kernel)
        for T in (Float32, Float64), dim in (1,2,3)
            Random.seed!(1234)
            mp = MPValue(Vec{dim,T}, it)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            l = 0.5*spacing(mesh) / 2
            @test all(1:100) do _
                x = rand(Vec{dim, T})
                update!(mp, (;x,l), mesh)
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                PU && LFR
            end
        end
    end

    @testset "Positivity condition in kernel correction" begin
        function kernelvalue(mp, xp, mesh, i)
            fillzero!(mp.w)
            update!(mp, xp, mesh)
            j = findfirst(==(i), neighboringnodes(mp))
            j === nothing ? zero(eltype(mp.w)) : mp.w[j]
        end
        function kernelvalues(mesh::CartesianMesh{dim, T}, kernel, poly, index::CartesianIndex{dim}) where {dim, T}
            mp = MPValue(Vec{dim}, KernelCorrection(kernel, poly))
            L = kernel isa QuadraticBSpline ? 1.5 :
                kernel isa CubicBSpline     ? 2.0 : error()
            X = ntuple(i -> range(max(mesh[1][i],index[i]-L-1), min(mesh[end][i],index[i]+L-1)-sqrt(eps(T)), step=0.04), Val(dim))
            Z = Array{Float64}(undef, length.(X))
            for i in CartesianIndices(Z)
                @inbounds Z[i] = kernelvalue(mp, Vec(map(getindex, X, Tuple(i))), mesh, index)
            end
            Z
        end
        function ispositive(x)
            tol = sqrt(eps(typeof(x)))
            x > -tol
        end
        @testset "QuadraticBSpline" begin
            kern = QuadraticBSpline()
            lin = Sequoia.LinearPolynomial()
            multilin = Sequoia.MultiLinearPolynomial()
            @testset "2D" begin
                # boundaries
                mesh = CartesianMesh(1, (0,10), (0,10))
                for i in CartesianIndices((3,3))
                    if i == CartesianIndex(2,2)
                        @test !all(ispositive, kernelvalues(mesh, kern, lin, i))
                    else
                        @test all(ispositive, kernelvalues(mesh, kern, lin, i))
                    end
                    @test all(ispositive, kernelvalues(mesh, kern, multilin, i))
                end
                # greedy kernel correction (only for hyperrectangle)
                for I in CartesianIndices((3,3))
                    @test !all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2])), kern, lin,      i)), CartesianIndices(Tuple(I)))
                    @test  all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2])), kern, multilin, i)), CartesianIndices(Tuple(I)))
                end
            end
            @testset "3D" begin
                # for boundaries
                mesh = CartesianMesh(1, (0,10), (0,10), (0,10))
                for i in CartesianIndices((3,3,3))
                    if length(findall(==(2), Tuple(i))) > 1
                        @test !all(ispositive, kernelvalues(mesh, kern, lin, i))
                    else
                        @test all(ispositive, kernelvalues(mesh, kern, lin, i))
                    end
                    @test all(ispositive, kernelvalues(mesh, kern, multilin, i))
                end
                # greedy kernel correction (only for hyperrectangle)
                for I in CartesianIndices((3,3,3))
                    @test !all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3])), kern, lin,      i)), CartesianIndices(Tuple(I)))
                    @test  all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3])), kern, multilin, i)), CartesianIndices(Tuple(I)))
                end
            end
        end
        @testset "CubicBSpline" begin
            kern = CubicBSpline()
            lin = Sequoia.LinearPolynomial()
            multilin = Sequoia.MultiLinearPolynomial()
            @testset "2D" begin
                # for boundaries
                mesh = CartesianMesh(1, (0,10), (0,10))
                for i in CartesianIndices((4,4))
                    if length(findall(==(3), Tuple(i))) > 0
                        @test !all(ispositive, kernelvalues(mesh, kern, lin,      i))
                        @test !all(ispositive, kernelvalues(mesh, kern, multilin, i))
                    else
                        if i == CartesianIndex(2,2)
                            @test !all(ispositive, kernelvalues(mesh, kern, lin, i))
                        else
                            @test all(ispositive, kernelvalues(mesh, kern, lin, i))
                        end
                        @test all(ispositive, kernelvalues(mesh, kern, multilin, i))
                    end
                end
                # greedy kernel correction (only for hyperrectangle)
                for I in CartesianIndices((4,4))
                    @test !all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2])), kern, lin, i)), CartesianIndices(Tuple(I)))
                    if I == CartesianIndex(1,1)
                        @test all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2])), kern, multilin, i)), CartesianIndices(Tuple(I)))
                    else
                        @test !all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2])), kern, multilin, i)), CartesianIndices(Tuple(I)))
                    end
                end
            end
            @testset "3D" begin
                # for boundaries
                mesh = CartesianMesh(1, (0,10), (0,10), (0,10))
                for i in CartesianIndices((4,4,4))
                    if length(findall(==(3), Tuple(i))) > 0
                        @test !all(ispositive, kernelvalues(mesh, kern, lin,      i))
                        @test !all(ispositive, kernelvalues(mesh, kern, multilin, i))
                    else
                        if length(findall(==(2), Tuple(i))) > 1
                            @test !all(ispositive, kernelvalues(mesh, kern, lin, i))
                        else
                            @test all(ispositive, kernelvalues(mesh, kern, lin, i))
                        end
                        @test all(ispositive, kernelvalues(mesh, kern, multilin, i))
                    end
                end
                # greedy kernel correction (only for hyperrectangle)
                for I in CartesianIndices((4,4,4))
                    @test !all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3])), kern, lin, i)), CartesianIndices(Tuple(I)))
                    if I == CartesianIndex(1,1,1)
                        @test all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3])), kern, multilin, i)), CartesianIndices(Tuple(I)))
                    else
                        @test !all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3])), kern, multilin, i)), CartesianIndices(Tuple(I)))
                    end
                end
            end
        end
    end
end
