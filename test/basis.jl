@testset "BasisWeight" begin

function check_weight_layout(bw::Union{BasisWeight, Tesserae.BasisWeightArray}, ::Type{T}, ::Val{dim}, derivative) where {T, dim}
    nd = ifelse(bw isa BasisWeight, dim, dim+1)
    if derivative isa Order{0}
        @test propertynames(bw) === (:w,)
        @test hasproperty(bw, :w) && bw.w isa AbstractArray{T}
        @test !hasproperty(bw, :∇w)
        @test !hasproperty(bw, :∇²w)
        @test ndims(bw.w) == nd
    elseif derivative isa Order{1}
        @test propertynames(bw) === (:w, :∇w)
        @test hasproperty(bw, :w)  && bw.w  isa AbstractArray{T}
        @test hasproperty(bw, :∇w) && bw.∇w isa AbstractArray{Vec{dim,T}}
        @test !hasproperty(bw, :∇²w)
        @test size(bw.w) == size(bw.∇w)
        @test ndims(bw.w) == ndims(bw.∇w) == nd
    elseif derivative isa Order{2}
        @test propertynames(bw) === (:w, :∇w, :∇²w)
        @test hasproperty(bw, :w)   && bw.w   isa AbstractArray{T}
        @test hasproperty(bw, :∇w)  && bw.∇w  isa AbstractArray{Vec{dim,T}}
        @test hasproperty(bw, :∇²w) && bw.∇²w isa AbstractArray{<: SymmetricSecondOrderTensor{dim,T}}
        @test size(bw.w) == size(bw.∇w) == size(bw.∇²w)
        @test ndims(bw.w) == ndims(bw.∇w) == ndims(bw.∇²w) == nd
    elseif derivative isa Order{3}
        @test propertynames(bw) === (:w, :∇w, :∇²w, :∇³w)
        @test hasproperty(bw, :w)   && bw.w   isa AbstractArray{T}
        @test hasproperty(bw, :∇w)  && bw.∇w  isa AbstractArray{Vec{dim,T}}
        @test hasproperty(bw, :∇²w) && bw.∇²w isa AbstractArray{<: SymmetricSecondOrderTensor{dim,T}}
        @test hasproperty(bw, :∇³w) && bw.∇³w isa AbstractArray{<: Tensor{Tuple{@Symmetry{dim,dim,dim}},T}}
        @test size(bw.w) == size(bw.∇w) == size(bw.∇²w) == size(bw.∇³w)
        @test ndims(bw.w) == ndims(bw.∇w) == ndims(bw.∇²w) == ndims(bw.∇³w) == nd
    else
        error()
    end
end

@testset "CartesianMesh layout" begin
    basis = BSpline(Quadratic())
    n = 2
    for dim in (1,2,3)
        mesh = CartesianMesh(1, ntuple(d->(0,10), dim)...)
        for T in (Float32, Float64)
            bw = @inferred BasisWeight(T, basis, mesh)
            @test Tesserae.basis(bw) === basis
            @test bw.w isa Array{T}
            @test bw.∇w isa Array{Vec{dim,T}}
            @test ndims(bw.w) == dim
            @test ndims(bw.∇w) == dim
            @test size(bw.w) == size(bw.∇w)
            @test typeof(supportnodes(bw)) === CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}
            for order in 0:3
                derivative = Order(order)
                check_weight_layout((@inferred BasisWeight(T, basis, mesh; derivative)), T, Val(dim), derivative)
            end

            weights = @inferred generate_basis_weights(T, basis, mesh, n)
            @test size(weights) === (n,)
            @test Tesserae.basis(weights) === basis
            @test weights.w isa Array{T}
            @test weights.∇w isa Array{Vec{dim,T}}
            @test ndims(weights.w) == dim+1
            @test ndims(weights.∇w) == dim+1
            @test size(weights.w) === size(weights.∇w)
            @test typeof(weights[begin]) === eltype(weights)
            @test typeof(weights[end]) === eltype(weights)
            for order in 0:3
                derivative = Order(order)
                weights = @inferred generate_basis_weights(T, basis, mesh, n; derivative)
                check_weight_layout(weights, T, Val(dim), derivative)
                check_weight_layout(weights[begin], T, Val(dim), derivative)
                check_weight_layout(weights[end], T, Val(dim), derivative)
            end
        end
    end
end

@testset "Basis wrappers" begin
    mesh = CartesianMesh(1, (0,10), (0,10))
    T = Float64
    n = 2
    for kernel in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()), uGIMP())
        for extension in (identity, WLS, KernelCorrection)
            basis = extension(kernel)
            bw = @inferred BasisWeight(T, basis, mesh)
            @test Tesserae.basis(bw) === basis
            check_weight_layout(bw, T, Val(2), Order(1))
            weights = @inferred generate_basis_weights(T, basis, mesh, n)
            @test Tesserae.basis(weights) === basis
            check_weight_layout(weights, T, Val(2), Order(1))
            @test typeof(weights[begin]) === eltype(weights)
            @test typeof(weights[end]) === eltype(weights)
        end
    end
end

end # BasisWeight

@testset "Basis functions" begin

@testset "Check `update!` for `BasisWeight`" begin
    isapproxzero(x) = x + ones(x) ≈ ones(x)
    interior_point(::Val{dim}) where {dim} = Vec{dim}(i -> 0.45 + 0.01i)
    boundary_point(::Val{dim}) where {dim} = Vec{dim}(i -> i == 1 ? 0.02 : 0.45 + 0.01i)
    is_support_truncated(bw) = size(bw.w) != size(supportnodes(bw))

    function check_partition_of_unity(bw, x; atol=sqrt(eps(eltype(bw.w))))
        indices = supportnodes(bw)
        CI = CartesianIndices(indices) # local indices
        isapprox(sum(bw.w[CI]), 1) && isapproxzero(sum(bw.∇w[CI]))
    end
    function check_linear_field_reproduction(bw, x, X)
        indices = supportnodes(bw)
        CI = CartesianIndices(indices) # local indices
        isapprox(mapreduce((j,i) -> X[i]*bw.w[j],  +, CI, indices), x) &&
        isapprox(mapreduce((j,i) -> X[i]⊗bw.∇w[j], +, CI, indices), I)
    end
    function check_update!(bw, pt, x, mesh; partition=true, reproduces_linear=true, truncated=nothing)
        update!(bw, pt, mesh)
        @test !isempty(supportnodes(bw))
        if truncated !== nothing
            @test is_support_truncated(bw) === truncated
        end
        PU = check_partition_of_unity(bw, x)
        LFR = check_linear_field_reproduction(bw, x, mesh)
        @test (partition ? PU : !PU)
        @test (reproduces_linear ? LFR : !LFR)
    end

    @testset "$spline" for spline in (BSpline(Constant()), BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()))
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            bw = BasisWeight(spline, mesh)
            x = interior_point(Val(dim))
            check_update!(bw, x, x, mesh;
                          partition=true,
                          reproduces_linear=!(spline isa BSpline{Constant}),
                          truncated=false)
            if spline isa Union{BSpline{Quadratic}, BSpline{Cubic}}
                x = boundary_point(Val(dim))
                check_update!(bw, x, x, mesh; partition=false, reproduces_linear=false, truncated=true)
            end
        end
    end

    @testset "$spline" for spline in (SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()))
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            bw = BasisWeight(spline, mesh)
            x = interior_point(Val(dim))
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=false)
            if spline isa Union{SteffenBSpline{Quadratic}, SteffenBSpline{Cubic}}
                x = boundary_point(Val(dim))
                check_update!(bw, x, x, mesh; partition=true, reproduces_linear=false, truncated=true)
            end
        end
    end

    @testset "uGIMP()" begin
        gimp = uGIMP()
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            bw = BasisWeight(gimp, mesh)
            l = 0.5*spacing(mesh)
            x = interior_point(Val(dim))
            check_update!(bw, (;x,l), x, mesh; partition=true, reproduces_linear=true, truncated=false)
            x = boundary_point(Val(dim))
            check_update!(bw, (;x,l), x, mesh; partition=false, reproduces_linear=false, truncated=true)
        end
    end

    @testset "CPDI" begin
        cpdi = CPDI()
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            bw = BasisWeight(cpdi, mesh)
            l = 0.5*spacing(mesh)
            F = one(Mat{dim,dim})
            x = interior_point(Val(dim))
            check_update!(bw, (;x,l,F), x, mesh; partition=true, reproduces_linear=true)

            GridProp = NamedTuple{(:x, :m), Tuple{Vec{dim, Float64}, Float64}}
            spgrid = generate_grid(SpArray, GridProp, mesh)
            err = try
                supportnodes(bw, spgrid)
                nothing
            catch err
                sprint(showerror, err)
            end
            @test err isa String && occursin("CPDI is currently supported only on dense Grid, not SpGrid", err)
        end
    end

    @testset "WLS branches" begin
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            l = 0.5*spacing(mesh) / 2
            x = interior_point(Val(dim))

            bw = BasisWeight(WLS(BSpline(Quadratic())), mesh)
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=false)

            bw = BasisWeight(WLS(BSpline(Linear())), mesh)
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=false)

            bw = BasisWeight(WLS(BSpline(Quadratic()), Polynomial(MultiLinear())), mesh)
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=false)

            x = boundary_point(Val(dim))
            bw = BasisWeight(WLS(BSpline(Quadratic())), mesh)
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=true)

            bw = BasisWeight(WLS(uGIMP()), mesh)
            check_update!(bw, (;x,l), x, mesh; partition=true, reproduces_linear=true, truncated=true)
        end
    end

    @testset "KernelCorrection branches" begin
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            l = 0.5*spacing(mesh) / 2

            x = interior_point(Val(dim))
            bw = BasisWeight(KernelCorrection(BSpline(Quadratic())), mesh)
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=false)

            x = boundary_point(Val(dim))
            bw = BasisWeight(KernelCorrection(BSpline(Quadratic())), mesh)
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=true)

            bw = BasisWeight(KernelCorrection(uGIMP()), mesh)
            check_update!(bw, (;x,l), x, mesh; partition=true, reproduces_linear=true, truncated=true)
        end
    end

    @testset "$(Wrapper(kernel)) coverage" for Wrapper in (WLS, KernelCorrection),
                                                  kernel in (BSpline(Cubic()), BSpline(Tesserae.Quartic()), BSpline(Tesserae.Quintic()), SteffenBSpline(Linear()), SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()))
        basis = Wrapper(kernel)
        mesh = CartesianMesh(0.1, (0,1), (0,1))
        bw = BasisWeight(basis, mesh)
        x = interior_point(Val(2))
        check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=false)

        if kernel isa Union{BSpline{Cubic}, BSpline{Tesserae.Quartic}, BSpline{Tesserae.Quintic}, SteffenBSpline{Quadratic}, SteffenBSpline{Cubic}}
            x = boundary_point(Val(2))
            bw = BasisWeight(basis, mesh)
            check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=true)
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
            bw = BasisWeight(spline, mesh; derivative=Order(k))
            update!(bw, xp, mesh)
            nodeindices = supportnodes(bw)
            for ip in eachindex(nodeindices)
                i = nodeindices[ip]
                vals = Tesserae.basis_jet(Order(k), spline, xp, mesh, i)
                for a in 0:k
                    @test values(bw, a+1)[ip] ≈ vals[a+1] atol=sqrt(eps(Float64))
                end
            end
        end
    end
end

@testset "Positivity in kernel correction" begin
    function kernelvalue(bw, xp, mesh, i)
        fillzero!(bw.w)
        update!(bw, xp, mesh)
        j = findfirst(==(i), supportnodes(bw))
        j === nothing ? zero(eltype(bw.w)) : bw.w[j]
    end
    function kernelvalues(mesh::CartesianMesh{dim}, kernel, poly, index::CartesianIndex{dim}) where {dim}
        bw = BasisWeight(KernelCorrection(kernel, poly), mesh)
        L = kernel isa BSpline{Quadratic} ? 1.5 :
            kernel isa BSpline{Cubic}     ? 2.0 : error()
        X = ntuple(i -> range(max(mesh[1][i],index[i]-L-1), min(mesh[end][i],index[i]+L-1)-sqrt(eps(Float64)), step=1/11), Val(dim)) # 1/10 is too coarse for checking
        Z = Array{Float64}(undef, length.(X))
        for i in CartesianIndices(Z)
            @inbounds Z[i] = kernelvalue(bw, Vec(map(getindex, X, Tuple(i))), mesh, index)
        end
        Z
    end
    function ispositive(x)
        tol = sqrt(eps(typeof(x)))
        x > -tol
    end
    is_positive_everywhere(mesh, kernel, poly, index) =
        all(ispositive, kernelvalues(mesh, kernel, poly, index))

    @testset "Quadratic B-spline" begin
        kern = BSpline(Quadratic())
        lin = Polynomial(Linear())
        multilin = Polynomial(MultiLinear())
        @testset "2D" begin
            mesh = CartesianMesh(1, (0,10), (0,10))
            @test is_positive_everywhere(mesh, kern, lin, CartesianIndex(1,1))
            @test !is_positive_everywhere(mesh, kern, lin, CartesianIndex(2,2))
            @test is_positive_everywhere(mesh, kern, multilin, CartesianIndex(2,2))

            for I in (CartesianIndex(1,1), CartesianIndex(3,2))
                domain = CartesianMesh(1, (0,I[1]), (0,I[2]))
                @test !all(i -> is_positive_everywhere(domain, kern, lin, i), CartesianIndices(Tuple(I)))
                @test  all(i -> is_positive_everywhere(domain, kern, multilin, i), CartesianIndices(Tuple(I)))
            end
        end
        @testset "3D" begin
            mesh = CartesianMesh(1, (0,10), (0,10), (0,10))
            @test is_positive_everywhere(mesh, kern, lin, CartesianIndex(1,1,1))
            @test !is_positive_everywhere(mesh, kern, lin, CartesianIndex(2,2,1))
            @test is_positive_everywhere(mesh, kern, multilin, CartesianIndex(2,2,1))

            for I in (CartesianIndex(1,1,1), CartesianIndex(3,2,1))
                domain = CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3]))
                @test !all(i -> is_positive_everywhere(domain, kern, lin, i), CartesianIndices(Tuple(I)))
                @test  all(i -> is_positive_everywhere(domain, kern, multilin, i), CartesianIndices(Tuple(I)))
            end
        end
    end
    @testset "Cubic B-spline" begin
        kern = BSpline(Cubic())
        lin = Polynomial(Linear())
        multilin = Polynomial(MultiLinear())
        @testset "2D" begin
            mesh = CartesianMesh(1, (0,10), (0,10))
            @test is_positive_everywhere(mesh, kern, lin, CartesianIndex(1,1))
            @test !is_positive_everywhere(mesh, kern, lin, CartesianIndex(2,2))
            @test is_positive_everywhere(mesh, kern, multilin, CartesianIndex(2,2))
            @test !is_positive_everywhere(mesh, kern, lin, CartesianIndex(3,1))
            @test !is_positive_everywhere(mesh, kern, multilin, CartesianIndex(3,1))

            for (I, multilin_positive) in ((CartesianIndex(1,1), true), (CartesianIndex(2,1), false))
                domain = CartesianMesh(1, (0,I[1]), (0,I[2]))
                @test !all(i -> is_positive_everywhere(domain, kern, lin, i), CartesianIndices(Tuple(I)))
                @test all(i -> is_positive_everywhere(domain, kern, multilin, i), CartesianIndices(Tuple(I))) === multilin_positive
            end
        end
        @testset "3D" begin
            mesh = CartesianMesh(1, (0,10), (0,10), (0,10))
            @test is_positive_everywhere(mesh, kern, lin, CartesianIndex(1,1,1))
            @test !is_positive_everywhere(mesh, kern, lin, CartesianIndex(2,2,1))
            @test is_positive_everywhere(mesh, kern, multilin, CartesianIndex(2,2,1))
            @test !is_positive_everywhere(mesh, kern, lin, CartesianIndex(3,1,1))
            @test !is_positive_everywhere(mesh, kern, multilin, CartesianIndex(3,1,1))

            for (I, multilin_positive) in ((CartesianIndex(1,1,1), true), (CartesianIndex(2,1,1), false))
                domain = CartesianMesh(1, (0,I[1]), (0,I[2]), (0,I[3]))
                @test !all(i -> is_positive_everywhere(domain, kern, lin, i), CartesianIndices(Tuple(I)))
                @test all(i -> is_positive_everywhere(domain, kern, multilin, i), CartesianIndices(Tuple(I))) === multilin_positive
            end
        end
    end
end

@testset "Polynomial" begin
    polynomial_point(::Type{T}, ::Val{dim}) where {T, dim} =
        Vec{dim, T}(i -> T(i) / T(dim + 2))

    exponents(::Polynomial{Linear}, ::Val{1}) = ((0,), (1,))
    exponents(::Polynomial{Linear}, ::Val{2}) = ((0,0), (1,0), (0,1))
    exponents(::Polynomial{Linear}, ::Val{3}) = ((0,0,0), (1,0,0), (0,1,0), (0,0,1))
    exponents(::Polynomial{Quadratic}, ::Val{1}) = ((0,), (1,), (2,))
    exponents(::Polynomial{Quadratic}, ::Val{2}) = ((0,0), (1,0), (0,1), (1,1), (2,0), (0,2))
    exponents(::Polynomial{Quadratic}, ::Val{3}) = ((0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (2,0,0), (0,2,0), (0,0,2))
    exponents(::Polynomial{MultiLinear}, ::Val{1}) = exponents(Polynomial(Linear()), Val(1))
    exponents(::Polynomial{MultiLinear}, ::Val{2}) = ((0,0), (1,0), (0,1), (1,1))
    exponents(::Polynomial{MultiLinear}, ::Val{3}) = ((0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (1,1,1))
    exponents(::Polynomial{Tesserae.MultiQuadratic}, ::Val{1}) = exponents(Polynomial(Quadratic()), Val(1))
    exponents(::Polynomial{Tesserae.MultiQuadratic}, ::Val{2}) = ((0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2))
    exponents(::Polynomial{Tesserae.MultiQuadratic}, ::Val{3}) = (
        (0,0,0),
        (1,0,0), (0,1,0), (0,0,1),
        (1,1,0), (0,1,1), (1,0,1), (1,1,1),
        (2,0,0), (0,2,0), (0,0,2),
        (2,1,0), (2,0,1), (2,1,1),
        (1,2,0), (0,2,1), (1,2,1),
        (1,0,2), (0,1,2), (1,1,2),
        (2,2,0), (0,2,2), (2,0,2),
        (2,2,1), (1,2,2), (2,1,2), (2,2,2),
    )

    function monomial_derivative(exp::NTuple{dim,Int}, x::Vec{dim,T}, dirs::Tuple) where {dim,T}
        powers = collect(exp)
        value = one(T)
        for d in dirs
            iszero(powers[d]) && return zero(T)
            value *= powers[d]
            powers[d] -= 1
        end
        for d in 1:dim
            value *= x[d]^powers[d]
        end
        value
    end

    function matches_polynomial_derivative(actual, exps, x)
        all(CartesianIndices(size(actual))) do I
            indices = Tuple(I)
            term = first(indices)
            dirs = indices[2:end]
            actual[indices...] ≈ monomial_derivative(exps[term], x, dirs)
        end
    end

    function check_polynomial(poly, ::Val{max_order}, ::Type{T}, ::Val{dim}; check_values=true) where {max_order,T,dim}
        x = polynomial_point(T, Val(dim))
        exps = exponents(poly, Val(dim))
        vals = values(Order(max_order), poly, x)
        @test all(v -> eltype(v) == T, vals)
        check_values || return

        for order in 0:max_order
            @test matches_polynomial_derivative(vals[order+1], exps, x)
        end
    end

    @testset "Linear" begin
        poly = Polynomial(Linear())
        for dim in (1,2,3)
            check_polynomial(poly, Val(4), Float64, Val(dim))
            check_polynomial(poly, Val(4), Float32, Val(dim); check_values=false)
        end
    end

    @testset "MultiLinear" begin
        poly = Polynomial(MultiLinear())
        for dim in (1,2,3)
            check_polynomial(poly, Val(4), Float64, Val(dim))
            check_polynomial(poly, Val(4), Float32, Val(dim); check_values=false)
        end
    end

    @testset "$poly" for poly in (Polynomial(Quadratic()), Polynomial(Tesserae.MultiQuadratic()))
        for dim in (1,2,3)
            check_polynomial(poly, Val(2), Float64, Val(dim))
            check_polynomial(poly, Val(2), Float32, Val(dim); check_values=false)
        end
    end
end

end # "Basis functions"
