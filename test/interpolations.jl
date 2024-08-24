@testset "MPValue" begin

@testset "CartesianMesh" begin
    for dim in (1,2,3)
        mesh = CartesianMesh(1, ntuple(d->(0,10), dim)...)
        for T in (Float32, Float64)
            for kernel in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()), uGIMP())
                for extension in (identity, WLS, KernelCorrection)
                    it = extension(kernel)

                    function check_mpvalue(mp::Union{MPValue, Tesserae.MPValueVector}, derivative)
                        if derivative isa Order{0}
                            @test hasproperty(mp, :w) && mp.w isa AbstractArray{T}
                            @test !hasproperty(mp, :∇w)
                            @test !hasproperty(mp, :∇∇w)
                            @test ndims(mp.w) == ifelse(mp isa MPValue, dim, dim+1)
                        elseif derivative isa Order{1}
                            @test hasproperty(mp, :w)  && mp.w  isa AbstractArray{T}
                            @test hasproperty(mp, :∇w) && mp.∇w isa AbstractArray{Vec{dim,T}}
                            @test !hasproperty(mp, :∇∇w)
                            @test size(mp.w) == size(mp.∇w)
                            @test ndims(mp.w) == ndims(mp.∇w) == ifelse(mp isa MPValue, dim, dim+1)
                        elseif derivative isa Order{2}
                            @test hasproperty(mp, :w)   && mp.w   isa AbstractArray{T}
                            @test hasproperty(mp, :∇w)  && mp.∇w  isa AbstractArray{Vec{dim,T}}
                            @test hasproperty(mp, :∇∇w) && mp.∇∇w isa AbstractArray{<: SymmetricSecondOrderTensor{dim,T}}
                            @test size(mp.w) == size(mp.∇w) == size(mp.∇∇w)
                            @test ndims(mp.w) == ndims(mp.∇w) == ndims(mp.∇∇w) == ifelse(mp isa MPValue, dim, dim+1)
                        elseif derivative isa Order{3}
                            @test hasproperty(mp, :w)    && mp.w    isa AbstractArray{T}
                            @test hasproperty(mp, :∇w)   && mp.∇w   isa AbstractArray{Vec{dim,T}}
                            @test hasproperty(mp, :∇∇w)  && mp.∇∇w  isa AbstractArray{<: SymmetricSecondOrderTensor{dim,T}}
                            @test hasproperty(mp, :∇∇∇w) && mp.∇∇∇w isa AbstractArray{<: Tensor{Tuple{@Symmetry{dim,dim,dim}},T}}
                            @test size(mp.w) == size(mp.∇w) == size(mp.∇∇w) == size(mp.∇∇∇w)
                            @test ndims(mp.w) == ndims(mp.∇w) == ndims(mp.∇∇w) == ndims(mp.∇∇∇w) == ifelse(mp isa MPValue, dim, dim+1)
                        else
                            error()
                        end
                    end

                    ###########
                    # MPValue #
                    ###########
                    mp = @inferred MPValue(T, it, mesh)
                    @test Tesserae.interpolation(mp) === it
                    @test mp.w isa Array{T}
                    @test mp.∇w isa Array{Vec{dim,T}}
                    @test ndims(mp.w) == dim
                    @test ndims(mp.∇w) == dim
                    @test size(mp.w) == size(mp.∇w)
                    @test typeof(neighboringnodes(mp)) === CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}
                    check_mpvalue((@inferred MPValue(T, it, mesh; derivative=Order(0))), Order(0))
                    check_mpvalue((@inferred MPValue(T, it, mesh; derivative=Order(1))), Order(1))
                    check_mpvalue((@inferred MPValue(T, it, mesh; derivative=Order(2))), Order(2))
                    check_mpvalue((@inferred MPValue(T, it, mesh; derivative=Order(3))), Order(3))

                    #################
                    # MPValueVector #
                    #################
                    n = 10
                    mpvalues = @inferred generate_mpvalues(T, it, mesh, n)
                    @test size(mpvalues) === (n,)
                    @test Tesserae.interpolation(mpvalues) === it
                    @test mpvalues.w isa Array{T}
                    @test mpvalues.∇w isa Array{Vec{dim,T}}
                    @test ndims(mpvalues.w) == dim+1
                    @test ndims(mpvalues.∇w) == dim+1
                    @test size(mpvalues.w) === size(mpvalues.∇w)
                    @test all(eachindex(mpvalues)) do i
                        typeof(mpvalues[i]) === eltype(mpvalues)
                    end
                    for order in 0:3
                        mpvalues = @inferred generate_mpvalues(T, it, mesh, n; derivative=Order(order))
                        check_mpvalue(mpvalues, Order(order))
                        foreach(mp -> check_mpvalue(mp, Order(order)), mpvalues)
                    end
                end
            end
        end
    end
end

end # MPValue

@testset "Interpolations" begin

@testset "Check `update!` for `MPValue`" begin
    isapproxzero(x) = x + ones(x) ≈ ones(x)
    function check_partition_of_unity(mp, x; atol=sqrt(eps(eltype(mp.w))))
        indices = neighboringnodes(mp)
        CI = CartesianIndices(indices) # local indices
        isapprox(sum(mp.w[CI]), 1) && isapproxzero(sum(mp.∇w[CI]))
    end
    function check_linear_field_reproduction(mp, x, X)
        indices = neighboringnodes(mp)
        CI = CartesianIndices(indices) # local indices
        isapprox(mapreduce((j,i) -> X[i]*mp.w[j],  +, CI, indices), x) &&
        isapprox(mapreduce((j,i) -> X[i]⊗mp.∇w[j], +, CI, indices), I)
    end

    @testset "$it" for it in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()))
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            mp = MPValue(it, mesh)
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(mp, x, mesh)
                isnearbounds = size(mp.w) != size(neighboringnodes(mp))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                isnearbounds ? (!PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "$it" for it in (SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()))
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            mp = MPValue(it, mesh)
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(mp, x, mesh)
                isnearbounds = size(mp.w) != size(neighboringnodes(mp))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                isnearbounds ? (PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "uGIMP()" begin
        it = uGIMP()
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            mp = MPValue(it, mesh)
            l = 0.5*spacing(mesh)
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(mp, (;x,l), mesh)
                isnearbounds = any(.!(l/2 .< x .< 1-l/2))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                # uGIMP doesn't have pertition of unity when very closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                isnearbounds ? (!PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "CPDI" begin
        it = CPDI()
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            mp = MPValue(it, mesh)
            l = 0.5*spacing(mesh)
            F = one(Mat{dim,dim})
            @test all(1:100) do _
                x = Vec{dim}(i -> l/2 + rand()*(1-l))
                update!(mp, (;x,l,F), mesh)
                # isnearbounds = any(.!(l/2 .< x .< 1-l/2))
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                # uGIMP doesn't have pertition of unity when very closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                # isnearbounds ? (!PU && !LFR) : (PU && LFR)
                PU && LFR
            end
        end
    end

    @testset "$(Wrapper(kernel))" for Wrapper in (WLS, KernelCorrection), kernel in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()), uGIMP())
        it = Wrapper(kernel)
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            mp = MPValue(it, mesh)
            l = 0.5*spacing(mesh) / 2
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(mp, (;x,l), mesh)
                PU = check_partition_of_unity(mp, x)
                LFR = check_linear_field_reproduction(mp, x, mesh)
                PU && LFR
            end
        end
    end
end

@testset "B-spline fast computation" begin
    # check by autodiff
    @testset "$it" for it in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()))
        for dim in (1,2,3)
            Random.seed!(1234)
            xp = rand(Vec{dim})
            # gradient
            grad = values(Order(1), it, xp)[end]
            grad_autodiff = gradient(xp -> Tensor(values(it, xp)), xp)
            for i in CartesianIndices(grad)
                for j in CartesianIndices(grad[i])
                    @test grad[i][j] ≈ grad_autodiff[i,j]
                end
            end
            # hessian
            hess = values(Order(2), it, xp)[end]
            hess_autodiff = hessian(xp -> Tensor(values(it, xp)), xp)
            for i in CartesianIndices(hess)
                for j in CartesianIndices(hess[i])
                    @test hess[i][j] ≈ hess_autodiff[i,j]
                end
            end
            # all
            al = values(Order(3), it, xp)[end]
            al_autodiff = gradient(xp -> hessian(xp -> Tensor(values(it, xp)), xp), xp)
            for i in CartesianIndices(al)
                for j in CartesianIndices(al[i])
                    @test al[i][j] ≈ al_autodiff[i,j]
                end
            end
        end
    end
end

@testset "Positivity in kernel correction" begin
    function kernelvalue(mp, xp, mesh, i)
        fillzero!(mp.w)
        update!(mp, xp, mesh)
        j = findfirst(==(i), neighboringnodes(mp))
        j === nothing ? zero(eltype(mp.w)) : mp.w[j]
    end
    function kernelvalues(mesh::CartesianMesh{dim}, kernel, poly, index::CartesianIndex{dim}) where {dim}
        mp = MPValue(KernelCorrection(kernel, poly), mesh)
        L = kernel isa BSpline{Quadratic} ? 1.5 :
            kernel isa BSpline{Cubic}     ? 2.0 : error()
        X = ntuple(i -> range(max(mesh[1][i],index[i]-L-1), min(mesh[end][i],index[i]+L-1)-sqrt(eps(Float64)), step=1/11), Val(dim)) # 1/10 is too coarse for checking
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
    @testset "Quadratic B-spline" begin
        kern = BSpline(Quadratic())
        lin = Polynomial(Linear())
        multilin = Polynomial(MultiLinear())
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
            # aggressive kernel correction (only for hyperrectangle)
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
            # aggressive kernel correction (only for hyperrectangle)
            for I in CartesianIndices((3,3,3))
                @test !all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3])), kern, lin,      i)), CartesianIndices(Tuple(I)))
                @test  all(i -> all(ispositive, kernelvalues(CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3])), kern, multilin, i)), CartesianIndices(Tuple(I)))
            end
        end
    end
    @testset "Cubic B-spline" begin
        kern = BSpline(Cubic())
        lin = Polynomial(Linear())
        multilin = Polynomial(MultiLinear())
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
            # aggressive kernel correction (only for hyperrectangle)
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
            # aggressive kernel correction (only for hyperrectangle)
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

end # "Interpolations"
