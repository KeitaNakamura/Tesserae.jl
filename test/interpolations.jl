@testset "InterpolationWeight" begin

@testset "CartesianMesh" begin
    for dim in (1,2,3)
        mesh = CartesianMesh(1, ntuple(d->(0,10), dim)...)
        for T in (Float32, Float64)
            for kernel in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()), uGIMP())
                for extension in (identity, WLS, KernelCorrection)
                    interp = extension(kernel)

                    function check_weight(iw::Union{InterpolationWeight, Tesserae.InterpolationWeightArray}, derivative)
                        if derivative isa Order{0}
                            @test hasproperty(iw, :w) && iw.w isa AbstractArray{T}
                            @test !hasproperty(iw, :∇w)
                            @test !hasproperty(iw, :∇∇w)
                            @test ndims(iw.w) == ifelse(iw isa InterpolationWeight, dim, dim+1)
                        elseif derivative isa Order{1}
                            @test hasproperty(iw, :w)  && iw.w  isa AbstractArray{T}
                            @test hasproperty(iw, :∇w) && iw.∇w isa AbstractArray{Vec{dim,T}}
                            @test !hasproperty(iw, :∇∇w)
                            @test size(iw.w) == size(iw.∇w)
                            @test ndims(iw.w) == ndims(iw.∇w) == ifelse(iw isa InterpolationWeight, dim, dim+1)
                        elseif derivative isa Order{2}
                            @test hasproperty(iw, :w)   && iw.w   isa AbstractArray{T}
                            @test hasproperty(iw, :∇w)  && iw.∇w  isa AbstractArray{Vec{dim,T}}
                            @test hasproperty(iw, :∇²w) && iw.∇²w isa AbstractArray{<: SymmetricSecondOrderTensor{dim,T}}
                            @test size(iw.w) == size(iw.∇w) == size(iw.∇²w)
                            @test ndims(iw.w) == ndims(iw.∇w) == ndims(iw.∇²w) == ifelse(iw isa InterpolationWeight, dim, dim+1)
                        elseif derivative isa Order{3}
                            @test hasproperty(iw, :w)   && iw.w   isa AbstractArray{T}
                            @test hasproperty(iw, :∇w)  && iw.∇w  isa AbstractArray{Vec{dim,T}}
                            @test hasproperty(iw, :∇²w) && iw.∇²w isa AbstractArray{<: SymmetricSecondOrderTensor{dim,T}}
                            @test hasproperty(iw, :∇³w) && iw.∇³w isa AbstractArray{<: Tensor{Tuple{@Symmetry{dim,dim,dim}},T}}
                            @test size(iw.w) == size(iw.∇w) == size(iw.∇²w) == size(iw.∇³w)
                            @test ndims(iw.w) == ndims(iw.∇w) == ndims(iw.∇²w) == ndims(iw.∇³w) == ifelse(iw isa InterpolationWeight, dim, dim+1)
                        else
                            error()
                        end
                    end

                    #######################
                    # InterpolationWeight #
                    #######################
                    iw = @inferred InterpolationWeight(T, interp, mesh)
                    @test Tesserae.interpolation(iw) === interp
                    @test iw.w isa Array{T}
                    @test iw.∇w isa Array{Vec{dim,T}}
                    @test ndims(iw.w) == dim
                    @test ndims(iw.∇w) == dim
                    @test size(iw.w) == size(iw.∇w)
                    @test typeof(neighboringnodes(iw)) === CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}
                    check_weight((@inferred InterpolationWeight(T, interp, mesh; derivative=Order(0))), Order(0))
                    check_weight((@inferred InterpolationWeight(T, interp, mesh; derivative=Order(1))), Order(1))
                    check_weight((@inferred InterpolationWeight(T, interp, mesh; derivative=Order(2))), Order(2))
                    check_weight((@inferred InterpolationWeight(T, interp, mesh; derivative=Order(3))), Order(3))

                    ############################
                    # InterpolationWeightArray #
                    ############################
                    n = 10
                    weights = @inferred generate_interpolation_weights(T, interp, mesh, n)
                    @test size(weights) === (n,)
                    @test Tesserae.interpolation(weights) === interp
                    @test weights.w isa Array{T}
                    @test weights.∇w isa Array{Vec{dim,T}}
                    @test ndims(weights.w) == dim+1
                    @test ndims(weights.∇w) == dim+1
                    @test size(weights.w) === size(weights.∇w)
                    @test all(eachindex(weights)) do i
                        typeof(weights[i]) === eltype(weights)
                    end
                    for order in 0:3
                        weights = @inferred generate_interpolation_weights(T, interp, mesh, n; derivative=Order(order))
                        check_weight(weights, Order(order))
                        foreach(iw -> check_weight(iw, Order(order)), weights)
                    end
                end
            end
        end
    end
end

end # InterpolationWeight

@testset "Interpolations" begin

@testset "Check `update!` for `InterpolationWeight`" begin
    isapproxzero(x) = x + ones(x) ≈ ones(x)
    function check_partition_of_unity(iw, x; atol=sqrt(eps(eltype(iw.w))))
        indices = neighboringnodes(iw)
        CI = CartesianIndices(indices) # local indices
        isapprox(sum(iw.w[CI]), 1) && isapproxzero(sum(iw.∇w[CI]))
    end
    function check_linear_field_reproduction(iw, x, X)
        indices = neighboringnodes(iw)
        CI = CartesianIndices(indices) # local indices
        isapprox(mapreduce((j,i) -> X[i]*iw.w[j],  +, CI, indices), x) &&
        isapprox(mapreduce((j,i) -> X[i]⊗iw.∇w[j], +, CI, indices), I)
    end

    @testset "$spline" for spline in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), BSpline(Quadratic()))
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            iw = InterpolationWeight(spline, mesh)
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(iw, x, mesh)
                is_support_truncated = size(iw.w) != size(neighboringnodes(iw))
                PU = check_partition_of_unity(iw, x)
                LFR = check_linear_field_reproduction(iw, x, mesh)
                is_support_truncated ? (!PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "$spline" for spline in (SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()))
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            iw = InterpolationWeight(spline, mesh)
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(iw, x, mesh)
                is_support_truncated = size(iw.w) != size(neighboringnodes(iw))
                PU = check_partition_of_unity(iw, x)
                LFR = check_linear_field_reproduction(iw, x, mesh)
                is_support_truncated ? (PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "uGIMP()" begin
        gimp = uGIMP()
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            iw = InterpolationWeight(gimp, mesh)
            l = 0.5*spacing(mesh)
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(iw, (;x,l), mesh)
                is_support_truncated = any(.!(l/2 .< x .< 1-l/2))
                PU = check_partition_of_unity(iw, x)
                LFR = check_linear_field_reproduction(iw, x, mesh)
                # uGIMP doesn't have pertition of unity when very closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                is_support_truncated ? (!PU && !LFR) : (PU && LFR)
            end
        end
    end

    @testset "CPDI" begin
        cpdi = CPDI()
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            iw = InterpolationWeight(cpdi, mesh)
            l = 0.5*spacing(mesh)
            F = one(Mat{dim,dim})
            @test all(1:100) do _
                x = Vec{dim}(i -> l/2 + rand()*(1-l))
                update!(iw, (;x,l,F), mesh)
                # is_support_truncated = any(.!(l/2 .< x .< 1-l/2))
                PU = check_partition_of_unity(iw, x)
                LFR = check_linear_field_reproduction(iw, x, mesh)
                # uGIMP doesn't have pertition of unity when very closed to boundaries
                # if we follow eq.40 in Bardenhagen (2004)
                # is_support_truncated ? (!PU && !LFR) : (PU && LFR)
                PU && LFR
            end
        end
    end

    @testset "$(Wrapper(kernel))" for Wrapper in (WLS, KernelCorrection), kernel in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), BSpline(Tesserae.Quartic()), BSpline(Tesserae.Quintic()), SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()), uGIMP())
        interp = Wrapper(kernel)
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            iw = InterpolationWeight(interp, mesh)
            l = 0.5*spacing(mesh) / 2
            @test all(1:100) do _
                x = rand(Vec{dim})
                update!(iw, (;x,l), mesh)
                PU = check_partition_of_unity(iw, x)
                LFR = check_linear_field_reproduction(iw, x, mesh)
                PU && LFR
            end
        end
    end
end

@testset "B-spline fast computation" begin
    # check by autodiff
    k = 5
    @testset "$spline" for spline in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), BSpline(Tesserae.Quartic()), BSpline(Tesserae.Quintic()))
        for dim in (1,2,3)
            Random.seed!(1234)
            mesh = CartesianMesh(0.1, ntuple(i->(-1,2), Val(dim))...)
            xp = rand(Vec{dim})
            iw = InterpolationWeight(spline, mesh; derivative=Order(k))
            update!(iw, xp, mesh)
            nodeindices = neighboringnodes(iw)
            for ip in eachindex(nodeindices)
                i = nodeindices[ip]
                vals = values(Order(k), spline, xp, mesh, i)
                for a in 0:k
                    @test values(iw, a+1)[ip] ≈ vals[a+1] atol=sqrt(eps(Float64))
                end
            end
        end
    end
end

@testset "Positivity in kernel correction" begin
    function kernelvalue(iw, xp, mesh, i)
        fillzero!(iw.w)
        update!(iw, xp, mesh)
        j = findfirst(==(i), neighboringnodes(iw))
        j === nothing ? zero(eltype(iw.w)) : iw.w[j]
    end
    function kernelvalues(mesh::CartesianMesh{dim}, kernel, poly, index::CartesianIndex{dim}) where {dim}
        iw = InterpolationWeight(KernelCorrection(kernel, poly), mesh)
        L = kernel isa BSpline{Quadratic} ? 1.5 :
            kernel isa BSpline{Cubic}     ? 2.0 : error()
        X = ntuple(i -> range(max(mesh[1][i],index[i]-L-1), min(mesh[end][i],index[i]+L-1)-sqrt(eps(Float64)), step=1/11), Val(dim)) # 1/10 is too coarse for checking
        Z = Array{Float64}(undef, length.(X))
        for i in CartesianIndices(Z)
            @inbounds Z[i] = kernelvalue(iw, Vec(map(getindex, X, Tuple(i))), mesh, index)
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

@testset "Polynomial" begin
    for T in (Float32, Float64)
        for dim in (1,2,3)
            for poly in (Polynomial(Linear()), Polynomial(MultiLinear()), Polynomial(Quadratic()), Polynomial(Tesserae.MultiQuadratic()))
                xp = rand(Vec{dim, T})
                if poly.degree isa Union{Linear, MultiLinear}
                    vals = values(Order(4), poly, xp)
                    @test all(v -> eltype(v) == T, vals)
                    @test vals[2] ≈ gradient(x -> only(values(Order(0), poly, x)), xp)
                    @test vals[3] ≈ gradient(x -> gradient(x -> only(values(Order(0), poly, x)), x), xp)
                    @test vals[4] ≈ gradient(x -> gradient(x -> gradient(x -> only(values(Order(0), poly, x)), x), x), xp)
                    @test vals[5] ≈ gradient(x -> gradient(x -> gradient(x -> gradient(x -> only(values(Order(0), poly, x)), x), x), x), xp)
                else
                    vals = values(Order(2), poly, xp)
                    @test all(v -> eltype(v) == T, vals)
                    @test vals[2] ≈ gradient(x -> only(values(Order(0), poly, x)), xp)
                    @test vals[3] ≈ gradient(x -> gradient(x -> only(values(Order(0), poly, x)), x), xp)
                end
            end
        end
    end
end

end # "Interpolations"
