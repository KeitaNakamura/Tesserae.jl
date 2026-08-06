struct TestLinearKernel <: Tesserae.Kernel end

Tesserae.support_width(::TestLinearKernel) = 2
Tesserae.supportnodes(::TestLinearKernel, pt, mesh::CartesianMesh) =
    Tesserae.supportnodes(BSpline(Linear()), pt, mesh)
Tesserae.nodal_basis_jet(order::Order, ::TestLinearKernel, pt, mesh::CartesianMesh, i) =
    Tesserae.nodal_basis_jet(order, BSpline(Linear()), pt, mesh, i)

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

@testset "Basis-value allocation" begin
    Prop = @NamedTuple{N::Float64, ψ::Vec{2,Float64}}
    zeros = @inferred Tesserae.basis_value_zeros(Prop, Val(2), Order(2))
    @test propertynames(zeros) === (:N, :∇N, :∇²N, :ψ)
    @test zeros === (; N=0.0, ∇N=zero(Vec{2,Float64}), ∇²N=zero(SymmetricSecondOrderTensor{2,Float64}), ψ=zero(Vec{2,Float64}))

    vals = @inferred Tesserae.allocate_basis_values(Prop, BSpline(Linear()), Val(2); derivative=Order(2))
    @test propertynames(vals) === propertynames(zeros)
    @test eltype(vals.N) === Float64
    @test eltype(vals.∇N) === Vec{2,Float64}
    @test eltype(vals.∇²N) <: SymmetricSecondOrderTensor{2,Float64}
    @test eltype(vals.ψ) === Vec{2,Float64}

    @test_throws ArgumentError Tesserae.basis_value_zeros(NamedTuple{(),Tuple{}}, Val(2), Order(1))
    @test_throws ArgumentError Tesserae.basis_value_zeros(@NamedTuple{w::Float64, ∇w::Vec{2,Float64}}, Val(2), Order(1))
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

@testset "Explicit derivative order" begin
    mesh = CartesianMesh(1, (0,10), (0,10))
    basis = BSpline(Linear())
    original = BasisWeight(basis, mesh; derivative=Order(1))
    ψ = fill(Vec(1.0, 2.0), size(original.w))
    vals = merge(getfield(original, :vals), (; ψ))
    bw = Tesserae.BasisWeight(basis, vals, getfield(original, :indices), Order(1))

    @test Tesserae.derivative_order(bw) isa Order{1}
    @test propertynames(bw) === (:w, :∇w, :ψ)
    update!(bw, Vec(2.2, 3.4), mesh)
    @test all(==(Vec(1.0, 2.0)), bw.ψ)
end

@testset "Property schema" begin
    mesh = CartesianMesh(1, (0,10), (0,10))
    basis = BSpline(Linear())
    Prop = @NamedTuple{N::Float32, ψ::Vec{2,Float64}}

    bw = @inferred BasisWeight(Prop, basis, mesh; derivative=Order(2))
    @test propertynames(bw) === (:N, :∇N, :∇²N, :ψ)
    @test eltype(bw.N) === Float32
    @test eltype(bw.∇N) === Vec{2,Float32}
    @test eltype(bw.∇²N) <: SymmetricSecondOrderTensor{2,Float32}
    @test eltype(bw.ψ) === Vec{2,Float64}
    bw.ψ .= Ref(Vec(1.0, 2.0))
    update!(bw, Vec(2.2, 3.4), mesh)
    @test all(==(Vec(1.0, 2.0)), bw.ψ)

    weights = @inferred generate_basis_weights(Prop, basis, mesh, 2; derivative=Order(1))
    @test propertynames(weights) === (:N, :∇N, :ψ)
    @test eltype(weights.N) === Float32
    @test eltype(weights.∇N) === Vec{2,Float32}
    @test eltype(weights.ψ) === Vec{2,Float64}
    @test Tesserae.derivative_order(weights) isa Order{1}
    @test Tesserae.derivative_order(first(weights)) isa Order{1}
end

@testset "BasisWeightArray views" begin
    mesh = CartesianMesh(1, (0,10), (0,10))
    basis = BSpline(Linear())
    weights = generate_basis_weights(basis, mesh, 4)
    weights.w .= reshape(collect(eachindex(weights.w)), size(weights.w))

    weights_view = @inferred view(weights, 2:3)
    @test weights_view isa Tesserae.BasisWeightArray
    @test size(weights_view) == (2,)
    @test Tesserae.basis(weights_view) === basis
    @test weights_view[1].w == weights[2].w
    @test weights_view[2].w == weights[3].w
    @test parent(weights_view.w) === weights.w
    @test parent(weights_view.∇w) === weights.∇w

    weights_view.w[1, 1, 1] = -1
    @test weights.w[1, 1, 2] == -1

    matrix_weights = generate_basis_weights(basis, mesh, 3, 4)
    matrix_view = @inferred view(matrix_weights, :, 2:3)
    @test size(matrix_view) == (3, 2)
    @test matrix_view[2, 1].w == matrix_weights[2, 2].w

    column_view = @inferred view(matrix_weights, :, 2)
    @test size(column_view) == (3,)
    @test column_view[2].w == matrix_weights[2, 2].w

    cartesian_view = @inferred view(matrix_weights, CartesianIndex(2, 3))
    @test size(cartesian_view) == ()
    @test cartesian_view[].w == matrix_weights[2, 3].w
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
    @testset "Steffen boundary correction with full storage" begin
        mesh = CartesianMesh(0.1, (0,0.4))
        for (spline, x) in ((SteffenBSpline(Quadratic()), Vec(0.05)),
                            (SteffenBSpline(Cubic()), Vec(0.1)))
            bw = BasisWeight(spline, mesh)
            update!(bw, x, mesh)
            indices = supportnodes(bw)
            @test size(indices) == size(bw.w)
            for ip in eachindex(indices)
                vals = Tesserae.nodal_basis_jet(Order(1), spline, x, mesh, indices[ip])
                @test bw.w[ip] ≈ vals[1]
                @test bw.∇w[ip] ≈ vals[2]
            end
        end
    end

    @testset "Supported B-spline degrees" begin
        mesh = CartesianMesh(0.1, (0,1))
        @test_throws ArgumentError BasisWeight(BSpline(Tesserae.Degree(6)), mesh)
        @test_throws ArgumentError BasisWeight(SteffenBSpline(Tesserae.Quartic()), mesh)
        for spline in (SteffenBSpline(Quadratic()), SteffenBSpline(Cubic()))
            @test (@inferred Tesserae.jet(Order(2), spline, 0, 1)) ==
                  Tesserae.jet(Order(2), spline, 0.0, 1)
        end
    end

    @testset "uGIMP()" begin
        gimp = uGIMP()
        @test Tesserae.jet(Order(1), BSpline(Quadratic()), 0) == (0.75, 0.0)
        @test Tesserae.jet(Order(1), gimp, 0, 1) == (0.75, 0.0)
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            bw = BasisWeight(gimp, mesh)
            l = 0.5*spacing(mesh)
            x = interior_point(Val(dim))
            check_update!(bw, (;x,l), x, mesh; partition=true, reproduces_linear=true)
            x = boundary_point(Val(dim))
            check_update!(bw, (;x,l), x, mesh; partition=false, reproduces_linear=false, truncated=true)

            # At the largest supported particle length, uGIMP still has at
            # most three support nodes per axis and fits its fixed storage.
            l = spacing(mesh)
            x = Vec{dim}(_ -> 0.44)
            indices = Tesserae.supportnodes(gimp, (;x,l), mesh)
            @test all(size(indices) .≤ size(bw.w))
            check_update!(bw, (;x,l), x, mesh; partition=true, reproduces_linear=true)
        end

        mesh = CartesianMesh(0.1, (0,1))
        bw = BasisWeight(gimp, mesh)
        check_update!(bw, (;x=Vec(0.44), l=0.0), Vec(0.44), mesh;
                      partition=true, reproduces_linear=true)
        @test_throws ArgumentError update!(bw, (;x=Vec(0.44), l=1.01spacing(mesh)), mesh)

        l = 0.5spacing(mesh)
        @test size(Tesserae.supportnodes(gimp, (;x=Vec(0.40), l), mesh)) == (3,)
        @test size(Tesserae.supportnodes(gimp, (;x=Vec(0.44), l), mesh)) == (2,)

        h = 0.003694869486948695
        mesh32 = CartesianMesh(Float32, h, (0,1); warn=false)
        @test eltype(mesh32) == Vec{1,Float32}
        l32 = spacing(mesh32)
        @test_nowarn Tesserae.supportnodes(gimp, (;x=Vec(100l32), l=l32), mesh32)
    end

    @testset "Kernel extension" begin
        mesh = CartesianMesh(0.1, (0,1), (0,1))
        kernel = TestLinearKernel()
        x = interior_point(Val(2))
        bw = BasisWeight(kernel, mesh)
        check_update!(bw, x, x, mesh; partition=true, reproduces_linear=true, truncated=false)
    end

    @testset "CPDI" begin
        cpdi = CPDI()
        @test cpdi isa Basis
        @test !(cpdi isa Tesserae.Kernel)
        @test_throws MethodError WLS(cpdi)
        @test_throws MethodError KernelCorrection(cpdi)
        for dim in (1,2,3)
            mesh = CartesianMesh(0.1, ntuple(i->(0,1), Val(dim))...)
            bw = BasisWeight(cpdi, mesh)
            @test_throws ArgumentError BasisWeight(cpdi, mesh; derivative=Order(0))
            l = 0.5*spacing(mesh)
            F = one(Mat{dim,dim})
            x = interior_point(Val(dim))
            filter = trues(size(mesh))
            filter[begin] = false
            @test_throws ArgumentError update!(bw, (;x,l,F), mesh, filter)
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

        mesh = CartesianMesh(0.1, (0,1), (0,1))
        l = 0.5spacing(mesh)
        F = one(Mat{2,2})
        particles = Tesserae.StructArray((
            x=[Vec(0.45,0.46), Vec(0.5,0.5)],
            l=fill(l, 2),
            F=[0.2F, F],
        ))
        weights = generate_basis_weights(cpdi, mesh, length(particles))
        @test update!(weights, particles, mesh) === weights
        @test map(length ∘ supportnodes, weights) == [4, 9]
        for p in eachindex(particles)
            particle = (;x=particles.x[p], l=particles.l[p], F=particles.F[p])
            scalar = BasisWeight(cpdi, mesh)
            update!(scalar, particle, mesh)
            @test collect(supportnodes(weights[p])) == collect(supportnodes(scalar))
            @test weights[p].w ≈ scalar.w
            @test weights[p].∇w ≈ scalar.∇w
            @test check_partition_of_unity(weights[p], particles.x[p])
            @test check_linear_field_reproduction(weights[p], particles.x[p], mesh)
        end

        filter = trues(size(mesh))
        filter[begin] = false
        @test_throws ArgumentError update!(weights, particles, mesh, filter)
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

    @testset "Filtered corrections" begin
        mesh = CartesianMesh(0.1, (0,1), (0,1))
        x = interior_point(Val(2))
        particles = Tesserae.StructArray((x=[x],))
        filter = trues(size(mesh))
        filter[first(Tesserae.supportnodes(BSpline(Quadratic()), x, mesh))] = false
        @test_throws ArgumentError update!(BasisWeight(BSpline(Quadratic()), mesh), x, mesh, filter)

        mixed_weights = BasisWeight[
            BasisWeight(WLS(BSpline(Quadratic())), mesh),
            BasisWeight(BSpline(Quadratic()), mesh),
        ]
        mixed_particles = Tesserae.StructArray((x=[x, x],))
        @test_throws ArgumentError update!(mixed_weights, mixed_particles, mesh, filter)

        empty_particles = Tesserae.StructArray((x=Vec{2,Float64}[],))
        empty_weights = generate_basis_weights(BSpline(Quadratic()), mesh, 0)
        @test update!(empty_weights, empty_particles, mesh, filter) === empty_weights

        for basis in (WLS(BSpline(Quadratic())), KernelCorrection(BSpline(Quadratic())))
            filter = trues(size(mesh))
            masked_node = first(Tesserae.supportnodes(basis, x, mesh))
            filter[masked_node] = false

            scalar = BasisWeight(basis, mesh)
            update!(scalar, x, mesh)
            update!(scalar, x, mesh, filter)
            @test check_partition_of_unity(scalar, x)
            @test check_linear_field_reproduction(scalar, x, mesh)
            masked_local_index = findfirst(==(masked_node), supportnodes(scalar))
            @test !isnothing(masked_local_index)
            @test iszero(scalar.w[masked_local_index])
            @test iszero(scalar.∇w[masked_local_index])

            weights = generate_basis_weights(basis, mesh, 1)
            update!(weights, particles, mesh)
            @test update!(weights, particles, mesh, filter) === weights
            @test supportnodes(weights[1]) == supportnodes(scalar)
            @test weights[1].w ≈ scalar.w
            @test weights[1].∇w ≈ scalar.∇w
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
                vals = @inferred Tesserae.nodal_basis_jet(Order(k), spline, xp, mesh, i)
                for a in 0:k
                    @test Tesserae.nodal_basis_values(bw, Order(a))[ip] ≈ vals[a+1] atol=sqrt(eps(Float64))
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
        vals = @inferred Tesserae.jet(Order(max_order), poly, x)
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
