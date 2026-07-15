iga_test_axes(degrees, knot_vectors) = map(Tesserae.NURBS.BSplineAxis, degrees, knot_vectors)
iga_test_control_net(degrees, knot_vectors, points, weights...) = Tesserae.NURBS.ControlNet(iga_test_axes(degrees, knot_vectors), points, weights...)
iga_test_degrees(net::Tesserae.NURBS.ControlNet) = map(axis -> axis.degree, net.axes)
iga_test_degrees(x) = Tesserae.degrees(x)
iga_test_knot_vectors(net::Tesserae.NURBS.ControlNet) = map(axis -> axis.knot_vector, net.axes)
iga_test_knot_vectors(patch::IGAPatch) = patch.knot_vectors

const nurbs_linear = Tesserae.NURBS.linear
const nurbs_quadratic = Tesserae.NURBS.quadratic
const nurbs_cubic = Tesserae.NURBS.cubic

@testset "IGA" begin
    # Shared Cartesian patch used by the basis, quadrature, assembly, and
    # particle-generation tests below.
    cmesh = CartesianMesh(0.25, (0,1), (0,1))
    mesh = IGAMesh(cmesh; degree=Quadratic())
    meshcells = collect(cells(mesh))
    basis = @inferred Tesserae.igabasis(mesh)
    qrule = generate_quadrature_rule(basis)
    patch = Tesserae.patches(mesh, 1)
    span = first(meshcells).span
    ξ = Tesserae.span_point(patch, span, first(qrule.points))

    @testset "Mesh" begin
        # CartesianMesh(0.25, (0,1), (0,1)) gives 4x4 active knot spans and
        # 6x6 quadratic control points.
        @test length(mesh) == 36
        @test Tesserae.ncells(mesh) == 16
        @test length(meshcells) == 16
        @test meshcells[1] == IGACell(1, 1, CartesianIndex(3,3))
        @test meshcells[end] == IGACell(16, 1, CartesianIndex(6,6))
        @test iga_test_degrees(patch) == (Quadratic(), Quadratic())
        @test typeof(basis) === IGABasis{2, Tuple{Quadratic, Quadratic}}
        @test iga_test_degrees(basis) == (Quadratic(), Quadratic())
        @test (@inferred Tesserae.nsupportnodes(basis)) === 9
        @test supportnodes(mesh) === mesh.used_controlpoint_ids
        @test supportnodes(mesh) == collect(eachindex(mesh))
        @test (@inferred supportnodes(mesh, meshcells[1])) === Tesserae.SVector(1, 2, 3, 7, 8, 9, 13, 14, 15)
        @test (@inferred supportnodes(mesh, meshcells[end])) === Tesserae.SVector(22, 23, 24, 28, 29, 30, 34, 35, 36)
    end

    @testset "Mesh conversion" begin
        # A single NURBS curve should become one IGA patch with identical
        # degree, knot vector, control points, and weights.
        curve_degrees = (nurbs_quadratic,)
        curve_knots = ([0.0,0.0,0.0,0.5,1.0,1.0,1.0],)
        curve_controlpoints = [Vec(Float64(i-1), 0.0) for i in 1:4]
        curve_weights = collect(1.0:4.0)
        curve_control = iga_test_control_net(curve_degrees, curve_knots, curve_controlpoints, curve_weights)
        curve_mesh = IGAMesh(curve_control)
        curve_patch = Tesserae.patches(curve_mesh, 1)
        @test iga_test_degrees(curve_patch) == (Quadratic(),)
        @test iga_test_knot_vectors(curve_patch) == curve_knots
        @test curve_patch.controlpoint_ids == [1, 2, 3, 4]
        @test curve_mesh.controlpoints == curve_controlpoints
        @test curve_mesh.weights == curve_weights

        # merge=true shares a global control-point id only when both the point
        # coordinate and rational weight match.
        duplicate_axis = Tesserae.NURBS.BSplineAxis(nurbs_linear, [0.0,0.0,1.0,1.0])
        duplicate_points = [Vec(0.0, 0.0), Vec(0.0, 0.0)]
        duplicate_control = Tesserae.NURBS.ControlNet((duplicate_axis,), duplicate_points, [1.0, 1.0])
        duplicate_mesh = IGAMesh(duplicate_control; merge=true)
        @test Tesserae.patches(duplicate_mesh, 1).controlpoint_ids == [1, 1]
        @test duplicate_mesh.controlpoints == [Vec(0.0, 0.0)]
        @test supportnodes(duplicate_mesh) == [1]
        weighted_duplicate = Tesserae.NURBS.ControlNet((duplicate_axis,), duplicate_points, [1.0, 2.0])
        weighted_duplicate_mesh = IGAMesh(weighted_duplicate; merge=true)
        @test Tesserae.patches(weighted_duplicate_mesh, 1).controlpoint_ids == [1, 2]

        # Merge tolerances are forwarded to both control-point and weight
        # comparisons.
        near_points = [Vec(1.0, 0.0), Vec(1.01, 0.0)]
        near_control = Tesserae.NURBS.ControlNet((duplicate_axis,), near_points, [1.0, 1.0])
        @test length(IGAMesh(near_control; merge=true)) == 2
        @test length(IGAMesh(near_control; merge=true, atol=0.02)) == 1
        @test length(IGAMesh(near_control; merge=true, rtol=0.02)) == 1
        @test length(IGAMesh(near_control; merge=true, atol=0.0, rtol=0.02)) == 1

        # Multi-patch conversion keeps separate patch connectivity, and
        # optionally shares interface control points.
        first_piece = Tesserae.NURBS.line(Vec(0.0, 0.0), Vec(1.0, 0.0))
        second_piece = Tesserae.NURBS.line(Vec(1.0, 0.0), Vec(2.0, 0.0))
        multi_mesh = IGAMesh([first_piece, second_piece])
        @test Tesserae.patches(multi_mesh, 1).controlpoint_ids == [1, 2]
        @test Tesserae.patches(multi_mesh, 2).controlpoint_ids == [3, 4]
        @test length(multi_mesh) == 4
        merged_multi_mesh = IGAMesh([first_piece, second_piece]; merge=true)
        @test Tesserae.patches(merged_multi_mesh, 1).controlpoint_ids == [1, 2]
        @test Tesserae.patches(merged_multi_mesh, 2).controlpoint_ids == [2, 3]
        @test length(merged_multi_mesh) == 3
        @test supportnodes(merged_multi_mesh) == [1, 2, 3]
        quadratic_piece = Tesserae.NURBS.arc(Vec(2.0, 0.0), 1.0, 0.0, π/2)
        @test_throws ArgumentError IGAMesh([first_piece, quadratic_piece])

        # A closed NURBS circle has duplicate endpoint control points; merging
        # should identify the seam.
        generated_circle = Tesserae.NURBS.circle(Vec(0.0, 0.0), 1.0)
        circle_mesh = IGAMesh(generated_circle; merge=true)
        circle_patch = Tesserae.patches(circle_mesh, 1)
        @test circle_patch.controlpoint_ids[begin] == circle_patch.controlpoint_ids[end]
        @test length(circle_mesh) == length(generated_circle.points) - 1

        # Surface conversion flattens tensor-product control points into the
        # global mesh storage while keeping tensor-product patch ids.
        degrees = (nurbs_quadratic, nurbs_quadratic)
        knots = ([0.0,0.0,0.0,0.5,1.0,1.0,1.0], [0.0,0.0,0.0,1/3,2/3,1.0,1.0,1.0])
        quad_controlpoints = map(CartesianIndices((4, 5))) do I
            Vec(Float64(I[1]-1), Float64(I[2]-1))
        end
        quad_control = iga_test_control_net(degrees, knots, quad_controlpoints)
        @test quad_control isa Tesserae.NURBS.ControlNet
        @test size(quad_control.points) == (4, 5)
        @test all(isone, quad_control.weights)
        @test quad_control.points[1,1] ≈ Vec(0.0, 0.0)
        @test quad_control.points[end,1] ≈ Vec(3.0, 0.0)
        @test quad_control.points[end,end] ≈ Vec(3.0, 4.0)
        @test quad_control.points[1,end] ≈ Vec(0.0, 4.0)
        quad_mesh = IGAMesh(quad_control)
        quad_patch = Tesserae.patches(quad_mesh, 1)
        @test iga_test_degrees(quad_patch) == (Quadratic(), Quadratic())
        @test iga_test_knot_vectors(quad_patch) == knots
        @test quad_patch.controlpoint_ids == Array(LinearIndices(quad_control.points))
        @test quad_mesh.controlpoints == vec(quad_control.points)
        @test quad_mesh.weights == vec(quad_control.weights)
    end

    @testset "Quadrature" begin
        # For degree p, IGA uses p+1 Gauss points per parametric direction.
        @test length(generate_quadrature_rule(IGABasis((Constant(),))).points) == 1
        @test length(qrule.points) == 9
        @test sum(qrule.weights) ≈ 4
        high_order_rule = @inferred generate_quadrature_rule(Float32, IGABasis((Tesserae.Degree(6), Quadratic())))
        @test length(high_order_rule.points) == 21
        @test eltype(high_order_rule.weights) === Float32
        @test sum(high_order_rule.weights) ≈ 4
        @test_throws ArgumentError generate_quadrature_rule(BigFloat, basis)
        anisotropic_rule = @inferred generate_quadrature_rule(IGABasis((Linear(), Quadratic())))
        @test length(anisotropic_rule.points) == 6
        @test sum(anisotropic_rule.weights) ≈ 4
        @test length(generate_quadrature_rule(IGABasis((Tesserae.Quintic(),))).points) == 6

        # Parent Gauss points are mapped from [-1,1]^dim to each knot span.
        α = √(3/5)
        @test Tesserae.span_point(patch, span, first(qrule.points)) ≈ Vec((1-α)/8, (1-α)/8)
        @test sum(qwt -> Tesserae.span_weight(patch, span, qwt), qrule.weights) ≈ 1/16
    end

    @testset "B-spline basis" begin
        knot_vector = iga_test_knot_vectors(patch)[1]
        x = ξ[1]

        # Active 1D B-splines form a partition of unity on the span.
        @test sum(Tesserae.cox_de_boor_values(Quadratic(), knot_vector, span[1], x)) ≈ 1
        @test sum(Tesserae.cox_de_boor_derivatives(Quadratic(), knot_vector, span[1], x)) ≈ 0

        # Degree-zero B-splines have one active value and zero derivative.
        N0, dN0 = @inferred Tesserae.cox_de_boor_values_and_derivatives(Constant(), [0.0, 1.0], 1, 0.5)
        @test N0 == Tesserae.SVector(1.0)
        @test dN0 == Tesserae.SVector(0.0)

        N, dN = @inferred Tesserae.cox_de_boor_values_and_derivatives(Quadratic(), knot_vector, span[1], x)
        @test N ≈ Tesserae.cox_de_boor_values(Quadratic(), knot_vector, span[1], x)
        @test dN ≈ Tesserae.cox_de_boor_derivatives(Quadratic(), knot_vector, span[1], x)

        cubic_knot_vector = [0.0,0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0,1.0]
        N3, dN3 = @inferred Tesserae.cox_de_boor_values_and_derivatives(Cubic(), cubic_knot_vector, 5, 0.375)
        @test N3 ≈ Tesserae.cox_de_boor_values(Cubic(), cubic_knot_vector, 5, 0.375)
        @test dN3 ≈ Tesserae.cox_de_boor_derivatives(Cubic(), cubic_knot_vector, 5, 0.375)
    end

    @testset "Tensor product basis" begin
        knot_vector = iga_test_knot_vectors(patch)[1]
        N̂, dNdξ = @inferred Tesserae.iga_basis_values_and_gradients(patch, span, ξ)
        N1, dN1 = Tesserae.cox_de_boor_values_and_derivatives(Quadratic(), knot_vector, span[1], ξ[1])
        N2, dN2 = Tesserae.cox_de_boor_values_and_derivatives(Quadratic(), knot_vector, span[2], ξ[2])

        # Tensor-product values are products of the 1D basis values.
        @test sum(N̂) ≈ 1
        @test isapprox(sum(dNdξ), Vec(0.0, 0.0); atol=1e-14)
        @test N̂[1] ≈ N1[1] * N2[1]
        @test N̂[2] ≈ N1[2] * N2[1]
        @test N̂[4] ≈ N1[1] * N2[2]
        @test dNdξ[1] ≈ Vec(dN1[1] * N2[1], N1[1] * dN2[1])
    end

    @testset "Rational basis" begin
        N̂, dNdξ = Tesserae.iga_basis_values_and_gradients(patch, span, ξ)

        # Unit control-point weights must recover the B-spline basis.
        R, dR = @inferred Tesserae.rational_basis_values_and_gradients(N̂, dNdξ, one.(N̂))
        @test R ≈ N̂
        @test dR ≈ dNdξ

        # Non-uniform weights give the standard rational basis and still sum to one.
        controlpoint_weights = Tesserae.SVector(1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)
        R, dR = @inferred Tesserae.rational_basis_values_and_gradients(N̂, dNdξ, controlpoint_weights)
        W = sum(N̂ .* controlpoint_weights)
        dW = sum(dNdξ .* controlpoint_weights)
        @test R[1] ≈ N̂[1] * controlpoint_weights[1] / W
        @test dR[1] ≈ controlpoint_weights[1] * (dNdξ[1] * W - N̂[1] * dW) / W^2
        @test sum(R) ≈ 1
        @test isapprox(sum(dR), Vec(0.0, 0.0); atol=1e-14)
    end

    @testset "Basis weights" begin
        # IGA basis-weight arrays should carry the same support-node layout as
        # direct mesh support queries.
        igaweights = @inferred generate_basis_weights(Float64, mesh, 2, Tesserae.ncells(mesh))
        @test size(igaweights) == (2, Tesserae.ncells(mesh))
        @test typeof(Tesserae.basis(igaweights)) === typeof(basis)
        @test size(igaweights[1, meshcells[end]].w) == (9,)
        @test size(igaweights[1, meshcells[end]].∇w) == (9,)
        @test length(supportnodes(igaweights[1, meshcells[end]])) == 9

        vals = (w=reshape(collect(1.0:16.0), 1, 16),)
        indices = reshape(collect(1:16), 1, 16)
        weights = Tesserae.BasisWeightArray(nothing, vals, indices)
        @test weights[1, meshcells[end]].w[] == 16
        @test supportnodes(weights[1, meshcells[end]]) == 16
    end

    @testset "Particles" begin
        PointProp = @NamedTuple{x::Vec{2,Float64}, V::Float64}

        # IGA particles are the physical images of the basis quadrature points.
        points = @inferred generate_particles(PointProp, mesh)
        indices = supportnodes(mesh, first(meshcells))
        N, _ = Tesserae.iga_basis_values_and_gradients(patch, span, ξ)
        @test size(points) == (length(qrule.points), Tesserae.ncells(mesh))
        @test points.x[1, first(meshcells)] ≈ sum(N .* mesh[indices])
        @test all(iszero, points.V)

        # Rational geometry uses rational basis values for the same mapping.
        rational_mesh = IGAMesh(cmesh; degree=Quadratic(), weights=range(1.0, 2.0; length=length(mesh)))
        rational_points = @inferred generate_particles(PointProp, rational_mesh)
        rational_cell = first(cells(rational_mesh))
        rational_patch = Tesserae.patches(rational_mesh, rational_cell.patch)
        rational_ξ = Tesserae.span_point(rational_patch, rational_cell.span, first(qrule.points))
        Nbs, dNbs = Tesserae.iga_basis_values_and_gradients(rational_patch, rational_cell.span, rational_ξ)
        rational_ids = supportnodes(rational_mesh, rational_cell)
        R, _ = Tesserae.rational_basis_values_and_gradients(Nbs, dNbs, rational_mesh.weights[rational_ids])
        @test rational_points.x[1, rational_cell] ≈ sum(R .* rational_mesh[rational_ids])
    end

    @testset "Quadrature update" begin
        points = generate_particles(@NamedTuple{x::Vec{2,Float64}}, mesh, qrule)
        feweights = generate_basis_weights(Float64, mesh, size(points))
        measure = zeros(Float64, size(feweights))
        @test (@inferred update!(feweights, points, mesh; measure)) === feweights
        @test supportnodes(feweights[1, meshcells[1]]) == supportnodes(mesh, meshcells[1])
        @test sum(feweights[1, meshcells[1]].w) ≈ 1
        @test isapprox(sum(feweights[1, meshcells[1]].∇w), Vec(0.0, 0.0); atol=1e-14)
        @test sum(measure[:, meshcells[1]]) ≈ 1/16
        @test sum(measure) ≈ 1

        reversed_patch = IGAPatch(iga_test_degrees(patch), map(copy, iga_test_knot_vectors(patch)), reverse(patch.controlpoint_ids))
        reversed_mesh = IGAMesh([reversed_patch], mesh.controlpoints)
        reversed_weights = generate_basis_weights(reversed_mesh, size(points))
        @test_throws ArgumentError update!(reversed_weights, points, mesh)

        # Rational updates should use rational basis values in both N and ∇N.
        rational_mesh = IGAMesh(cmesh; degree=Quadratic(), weights=range(1.0, 2.0; length=length(mesh)))
        rational_points = generate_particles(@NamedTuple{x::Vec{2,Float64}}, rational_mesh)
        rational_weights = generate_basis_weights(Float64, rational_mesh, size(rational_points))
        rational_measure = zeros(Float64, size(rational_weights))
        @test update!(rational_weights, rational_points, rational_mesh; measure=rational_measure) === rational_weights

        rational_cell = first(cells(rational_mesh))
        rational_patch = Tesserae.patches(rational_mesh, rational_cell.patch)
        rational_ξ = Tesserae.span_point(rational_patch, rational_cell.span, first(qrule.points))
        Nbs, dNbs = Tesserae.iga_basis_values_and_gradients(rational_patch, rational_cell.span, rational_ξ)
        rational_ids = supportnodes(rational_mesh, rational_cell)
        R, dR = Tesserae.rational_basis_values_and_gradients(Nbs, dNbs, rational_mesh.weights[rational_ids])
        J = sum(rational_mesh[rational_ids] .⊗ dR)
        @test rational_weights[1, rational_cell].w ≈ R
        @test rational_weights[1, rational_cell].∇w ≈ dR .⊡ Ref(inv(J))
        @test !(rational_weights[1, rational_cell].w ≈ Nbs)

        # Boundary IGA meshes are lower-dimensional patches that still assemble
        # into the original global control-point ids.
        boundary_patch = IGAPatch((Quadratic(),), (copy(iga_test_knot_vectors(patch)[1]),), Array(patch.controlpoint_ids[:,1]))
        boundary_mesh = IGAMesh([boundary_patch], mesh.controlpoints)
        @test supportnodes(boundary_mesh) == collect(patch.controlpoint_ids[:,1])
        boundary_points = generate_particles(@NamedTuple{x::Vec{2,Float64}}, boundary_mesh)
        boundary_weights = generate_basis_weights(boundary_mesh, size(boundary_points); name=Val(:N))
        measure = zeros(Float64, size(boundary_weights))
        normal = zeros(Vec{2, Float64}, size(boundary_weights))
        @test (@inferred update!(boundary_weights, boundary_points, boundary_mesh; measure, normal)) === boundary_weights
        @test sum(measure) ≈ 1
        @test all(n -> n ≈ Vec(0.0, -1.0), normal)
    end

    @testset "Sparse matrix" begin
        function stored_pattern(A)
            I, J, _ = Tesserae.SparseArrays.findnz(A)
            Set(zip(I, J))
        end

        @test_throws UndefKeywordError create_sparse_matrix(mesh)
        @test_throws UndefKeywordError create_sparse_matrix(basis, mesh)

        # The scalar matrix pattern must be exactly the union of each cell's
        # support-node pairings. Values are assembled later by @P2G_Matrix.
        scalar_pattern = Set{Tuple{Int,Int}}()
        for cell in cells(mesh)
            ids = supportnodes(mesh, cell)
            for j in ids, i in ids
                push!(scalar_pattern, (i, j))
            end
        end

        A = @inferred create_sparse_matrix(mesh; ndofs=1)
        B = @inferred create_sparse_matrix(basis, mesh; ndofs=1)
        @test size(A) == (length(mesh), length(mesh))
        @test stored_pattern(A) == scalar_pattern
        @test stored_pattern(B) == scalar_pattern

        # Mixed dof matrices should expand the same support pattern into dof
        # pairings without changing the underlying IGA connectivity.
        mixed_pattern = Set{Tuple{Int,Int}}()
        row_dofs = LinearIndices((2, length(mesh)))
        col_dofs = LinearIndices((1, length(mesh)))
        for cell in cells(mesh)
            ids = supportnodes(mesh, cell)
            for j in ids, i in ids, col in col_dofs[:,j], row in row_dofs[:,i]
                push!(mixed_pattern, (row, col))
            end
        end

        C = @inferred create_sparse_matrix(Float32, basis, mesh; ndofs=(2, 1))
        @test size(C) == (2 * length(mesh), length(mesh))
        @test eltype(C) === Float32
        @test stored_pattern(C) == mixed_pattern
    end

    @testset "Heat problem" begin
        # Smoke-test the full IGA path against the FEM path: particles,
        # basis weights, stiffness assembly, boundary dofs, and solve.
        GridProp = @NamedTuple{x::Vec{2,Float64}, u::Float64, f::Float64}
        PointProp = @NamedTuple{x::Vec{2,Float64}, V::Float64}

        function heat_norm(mesh)
            grid = generate_grid(GridProp, mesh)
            points = generate_particles(PointProp, mesh)
            weights = generate_basis_weights(mesh, size(points); name=Val(:N))
            update!(weights, points, mesh; measure=points.V)
            K = create_sparse_matrix(mesh; ndofs=1)
            dofmask = trues(1, size(grid)...)
            for i in eachindex(mesh)
                x = mesh[i]
                dofmask[1,i] = !(x[1] == -1 || x[1] == 1 || x[2] == -1 || x[2] == 1)
            end
            dofmap = DofMap(dofmask)

            @P2G grid=>i points=>p weights=>ip begin
                f[i] = @∑ N[ip] * V[p]
            end
            @P2G_Matrix grid=>(i,j) points=>p weights=>(ip,jp) begin
                K[i,j] = @∑ ∇N[ip] ⋅ ∇N[jp] * V[p]
            end

            dofmap(grid.u) .= Symmetric(extract(K, dofmap)) \ Array(dofmap(grid.f))
            norm(grid.u)
        end

        cmesh = CartesianMesh(0.1, (-1,1), (-1,1))
        fem = heat_norm(FEMesh(cmesh))
        @test fem ≈ 3.3077439126413104
        @test heat_norm(IGAMesh(cmesh; degree=Linear())) ≈ fem
        @test heat_norm(IGAMesh(cmesh; degree=Quadratic())) ≈ 3.322343871533971
        @test heat_norm(IGAMesh(cmesh; degree=Cubic())) ≈ 3.332908531024825
    end
end
