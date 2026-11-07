@testset "BasisWeightArray parent and view updates" begin
    mesh = CartesianMesh(1.0, (0,4))
    template = generate_particles(@NamedTuple{x::Vec{1,Float64}, m::Float64}, mesh; alg=GridSampling())[1:4]
    template.x .= [Vec(0.75), Vec(1.25), Vec(2.25), Vec(2.75)]
    template.m .= 1:4
    leftmask = trues(size(mesh)); leftmask[1] = false
    rightmask = trues(size(mesh)); rightmask[end] = false

    function scatter(weights, particles)
        grid = generate_grid(@NamedTuple{x::Vec{1,Float64}, m::Float64, g::Vec{1,Float64}}, mesh)
        @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            g[i] = @∑ ∇w[ip] * m[p]
        end
        (copy(grid.m), copy(grid.g))
    end
    function check_transfer(weights, reference, particles)
        got, expected = scatter(weights, particles), scatter(reference, particles)
        @test got[1] ≈ expected[1]
        @test got[2] ≈ expected[2]
    end
    function snapshot(weights)
        (; mode=Tesserae.isdeferred(weights), filter=Tesserae.deferred_filter(weights),
           indices=copy(Tesserae.supportnodes_storage(weights)),
           vals=Tesserae.storageless(weights) ? nothing : map(copy, getfield(weights, :vals)))
    end
    function check_unchanged(weights, before)
        @test Tesserae.isdeferred(weights) == before.mode
        @test Tesserae.deferred_filter(weights) === before.filter
        @test Tesserae.supportnodes_storage(weights) == before.indices
        isnothing(before.vals) || @test getfield(weights, :vals) == before.vals
    end

    @testset "$basis" for basis in (WLS(BSpline(Quadratic())), KernelCorrection(BSpline(Quadratic())))
        particles = deepcopy(template)
        reference = generate_basis_weights(basis, mesh, length(particles))

        @testset "stored partial updates" begin
            weights = generate_basis_weights(basis, mesh, length(particles))
            subset = [2,1]
            wv, pv = view(weights, subset), view(particles, subset)
            update!(weights, particles, mesh, leftmask)
            update!(reference, particles, mesh, leftmask)
            check_transfer(weights, reference, particles)
            @test Tesserae.deferred_filter(wv) === leftmask

            untouched = map(a -> copy(a[:,3:4]), getfield(weights, :vals))
            particles.x[1] += Vec(0.05)
            @test update!(wv, pv, mesh, rightmask; deferred=false) === wv
            # Independent element updates provide the expected mixed-filter data.
            for p in subset
                update!(reference[p], Tesserae.LazyRow(particles, p), mesh, rightmask)
            end
            check_transfer(weights, reference, particles)
            @test Tesserae.deferred_filter(weights) === leftmask
            @test Tesserae.deferred_filter(wv) === leftmask
            @test map(a -> a[:,3:4], getfield(weights, :vals)) == untouched
            @test !Tesserae.isdeferred(weights)

            # Omitting the filter inherits the parent's filter, including in stored mode.
            @test (@inferred update!(wv, pv, mesh)) === wv
            update!(reference, particles, mesh, leftmask)
            check_transfer(weights, reference, particles)
            before = snapshot(weights)
            @test_throws ArgumentError update!(wv, pv, mesh; deferred=true)
            check_unchanged(weights, before)
            @test_throws AssertionError update!(wv, pv, mesh, trues(2))
            check_unchanged(weights, before)
            @test_throws AssertionError update!(wv, particles, mesh)
            check_unchanged(weights, before)

            # Invalid parent requests must not replace the policy before validation.
            @test_throws AssertionError update!(weights, particles, mesh, trues(2); deferred=true)
            check_unchanged(weights, before)
        end

        @testset "inherited deferred policy, storageless=$storageless" for storageless in (false, true)
            weights = generate_basis_weights(basis, mesh, length(particles); deferred=storageless)
            # All views predate the parent's first filtered update.
            views = (
                (view(weights, 1:2), view(particles, 1:2), 1:2),
                (view(weights, [3,1]), view(particles, [3,1]), [3,1]),
                (view(view(weights, 2:4), 2:3), view(view(particles, 2:4), 2:3), 3:4),
                (view(weights, 1), view(particles, 1), 1:1),
                (view(weights, :), particles, 1:4),
                (view(weights, 1:0), view(particles, 1:0), 1:0),
            )
            for mask in (leftmask, rightmask)
                update!(weights, particles, mesh, mask; deferred=true)
                update!(reference, particles, mesh, mask)
                before = snapshot(weights)
                check_transfer(weights, reference, particles)
                for (wv, pv, subset) in views
                    @test Tesserae.isdeferred(wv)
                    @test Tesserae.deferred_filter(wv) === mask
                    @test (@inferred update!(wv, pv, mesh)) === wv
                    @test update!(wv, pv, mesh, mask; deferred=true) === wv
                    check_transfer(wv, view(reference, subset), pv)
                    @test_throws ArgumentError update!(wv, pv, mesh; deferred=false)
                    # Even an equal copy could later diverge from the owner's mask.
                    @test_throws ArgumentError update!(wv, pv, mesh, copy(mask))
                    @test_throws ArgumentError update!(wv, pv, mesh, mask === leftmask ? rightmask : leftmask)
                    check_unchanged(weights, before)
                end

                if storageless
                    @test_throws ErrorException update!(weights, particles, mesh, rightmask; deferred=false)
                    check_unchanged(weights, before)
                else
                    # The same existing views follow the switch back to stored mode.
                    update!(weights, particles, mesh, mask)
                    check_transfer(weights, reference, particles)
                    for (wv, pv, subset) in views
                        @test !Tesserae.isdeferred(wv)
                        @test update!(wv, pv, mesh) === wv
                        check_transfer(wv, view(reference, subset), pv)
                    end
                end
            end

            # Views also observe in-place edits to the same filter object.
            mutablemask = copy(leftmask)
            update!(weights, particles, mesh, mutablemask; deferred=true)
            mutablemask[1] = true
            update!(reference, particles, mesh)
            for (wv, pv, subset) in views
                @test Tesserae.deferred_filter(wv) === mutablemask
                check_transfer(wv, view(reference, subset), pv)
            end

            # Omission on the parent resets its filter to all nodes.
            update!(weights, particles, mesh; deferred=true)
            update!(reference, particles, mesh)
            @test isnothing(Tesserae.deferred_filter(weights))
            for (wv, pv, subset) in views
                @test isnothing(Tesserae.deferred_filter(wv))
                @test update!(wv, pv, mesh, Tesserae.Trues(size(mesh))) === wv
                check_transfer(wv, view(reference, subset), pv)
            end
        end

        @testset "multidimensional views" begin
            ps = reshape(deepcopy(particles), 2, 2)
            weights = generate_basis_weights(basis, mesh, size(ps))
            wv, pv = view(weights, :, 2), view(ps, :, 2)
            update!(weights, ps, mesh, leftmask; deferred=true)
            update!(reference, particles, mesh, leftmask)
            @test update!(wv, pv, mesh) === wv
            check_transfer(wv, view(reference, 3:4), pv)
            before = snapshot(weights)
            @test_throws ArgumentError update!(wv, pv, mesh, rightmask)
            check_unchanged(weights, before)
            update!(weights, ps, mesh, rightmask)
            @test !Tesserae.isdeferred(wv)
            @test update!(wv, pv, mesh) === wv
            update!(reference, particles, mesh, rightmask)
            check_transfer(wv, view(reference, 3:4), pv)
        end

        @testset "converted views have independent policy" begin
            weights = generate_basis_weights(basis, mesh, length(particles))
            wv, pv = view(weights, 1:2), view(particles, 1:2)
            update!(weights, particles, mesh, leftmask; deferred=true)
            @test cpu(wv) === wv
            @test_throws ArgumentError update!(cpu(wv), pv, mesh; deferred=false)
            copied = Tesserae.Adapt.adapt(Tesserae.CPUDevice(), wv)
            update!(copied, pv, mesh, rightmask)
            @test !Tesserae.isdeferred(copied)
            @test Tesserae.isdeferred(weights)
            @test Tesserae.deferred_filter(weights) === leftmask
            update!(reference, particles, mesh, rightmask)
            check_transfer(copied, view(reference, 1:2), pv)
            update!(weights, particles, mesh; deferred=true)
            @test Tesserae.deferred_filter(copied) === rightmask
        end

        @testset "Array and view filters, storageless=$storageless" for storageless in (false, true),
                mask in (collect(leftmask), view(copy(leftmask), :))
            weights = generate_basis_weights(basis, mesh, length(particles); deferred=storageless)
            update!(weights, particles, mesh, mask; deferred=true)
            wv, pv = view(weights, [3,1]), view(particles, [3,1])
            @test (@inferred update!(wv, pv, mesh)) === wv
            @test Tesserae.deferred_filter(wv) === mask
            update!(reference, particles, mesh, mask)
            check_transfer(wv, view(reference, [3,1]), pv)
            @test_throws ArgumentError update!(wv, pv, mesh, copy(mask))
            @test Tesserae.deferred_filter(weights) === mask
            if !storageless
                update!(weights, particles, mesh, mask)
                @test (@inferred update!(wv, pv, mesh)) === wv
                @test !Tesserae.isdeferred(wv)
                @test Tesserae.deferred_filter(weights) === mask
                check_transfer(wv, view(reference, [3,1]), pv)
            end
        end
    end
end

@testset "Stored and deferred property ordering" begin
    mesh = CartesianMesh(1.0, (0,6))
    particles = Tesserae.StructArray((x=[Vec(2.25)],))
    Prop = @NamedTuple{N::Float32, c::Float64}
    for derivative in (Order(0), Order(1), Order(2), Order(3)), mode in (:stored, :flagged, :storageless)
        reference = generate_basis_weights(Prop, BSpline(Cubic()), mesh, 1; derivative)
        update!(reference, particles, mesh)
        reference.c .= 7
        weights = generate_basis_weights(Prop, BSpline(Cubic()), mesh, 1; derivative, deferred=mode==:storageless)
        weights.c .= 7
        update!(weights, particles, mesh; deferred=mode!=:stored)
        for w in (weights, view(weights, [1]))
            selected = Tesserae.select_weights(identity, w)
            window = Tesserae.transfer_support_window(selected, particles, 1, mesh)
            for names in (propertynames(w), reverse(propertynames(w)), (:N,), (:c,), (:c,:N), ())
                columns = @inferred Tesserae.weight_columns(selected, Val(names), particles, 1, mesh, window)
                for ip in eachindex(window)
                    values = @inferred Tesserae.weight_node_values(selected, columns, Val(names), ip)
                    @test keys(values) == names
                    for name in names
                        @test getproperty(values, name) ≈ getproperty(reference[1], name)[ip]
                    end
                end
            end
            @test_throws ErrorException Tesserae.weight_columns(selected, Val((:missing,)), particles, 1, mesh, window)
            @test_throws ErrorException Tesserae.weight_node_values(selected, nothing, Val((:missing,)), 1)
        end
    end
end

@testset "Views of weights that cannot defer" begin
    mesh = CartesianMesh(1.0,(0,4))
    particles = Tesserae.StructArray((x=[Vec(0.75),Vec(2.25)],l=fill(0.5,2),F=fill(one(Mat{1,1}),2)))
    weights = generate_basis_weights(CPDI(),mesh,length(particles))
    update!(weights,particles,mesh)
    wv, pv = view(weights,[2,1]), view(particles,[2,1])
    particles.x[1] += Vec(0.125)
    @test (@inferred update!(wv,pv,mesh)) === wv
    for (w,p) in zip(wv,pv)
        nodes = supportnodes(w)
        @test sum(w.w[ip] for ip in eachindex(nodes)) ≈ 1
        @test sum(w.w[ip]*mesh[nodes[ip]] for ip in eachindex(nodes)) ≈ p.x
    end
    before = map(copy,getfield(weights,:vals))
    @test_throws ErrorException update!(wv,pv,mesh;deferred=true)
    @test_throws ErrorException update!(wv,pv,mesh,Tesserae.Trues(size(mesh));deferred=true)
    @test getfield(weights,:vals) == before
    @test !Tesserae.isdeferred(weights)
end
@testset "Transfer resolves evaluation policy once and shares custom storage" begin
    mesh = CartesianMesh(1.0, (0,4))
    particles = generate_particles(@NamedTuple{x::Vec{1,Float64}, m::Float32}, mesh; alg=GridSampling())[1:4]
    particles.x .= [Vec(0.75), Vec(1.25), Vec(2.25), Vec(2.75)]
    particles.m .= 1
    mask = trues(size(mesh)); mask[1] = false
    othermask = trues(size(mesh)); othermask[end] = false
    b = KernelCorrection(BSpline(Quadratic()))
    for storageless in (false, true)
        weights = generate_basis_weights(@NamedTuple{N::Float32, c::Float32}, b, mesh, length(particles); deferred=storageless)
        weights.c .= 2
        update!(weights, particles, mesh, mask; deferred=true)
        viewweights = view(weights, [3,1])
        vp = view(particles, [3,1])
        selected = Tesserae.select_weights(identity, viewweights)
        # Updating the owner changes future transfers, not this resolved policy.
        update!(weights, particles, mesh, othermask; deferred=true)
        @test Tesserae.deferred_filter(selected) === mask
        window = Tesserae.transfer_support_window(selected, vp, 1, mesh)
        columns = Tesserae.weight_columns(selected, Val((:N,:∇N,:c)), vp, 1, mesh, window)
        weights.c .= 3
        for ip in eachindex(window)
            values = @inferred Tesserae.weight_node_values(selected, columns, Val((:N,:∇N,:c)), ip)
            @test values.N isa Float32
            @test values.∇N isa Vec{1,Float32}
            @test values.c === 3f0
        end
        # A transfer remains in its selected mode if a particle equation updates
        # the owner; G2P's following P2G half must still use the original mask.
        function scatter(weights, ps, change)
            grid = generate_grid(@NamedTuple{x::Vec{1,Float64}, m::Float32}, mesh)
            @G2P2G grid=>i ps=>p weights=>ip begin
                x[p] += $(Vec(0.0)) * $change(p)
                m[i] = @∑ N[ip] * m[p]
            end
            copy(grid.m)
        end
        update!(weights, particles, mesh, mask; deferred=true)
        expected = scatter(viewweights, vp, _->0)
        change(p) = (update!(weights, particles, mesh, othermask; deferred=true); 0)
        @test scatter(viewweights, vp, change) ≈ expected
        if !storageless
            update!(weights, particles, mesh, mask; deferred=true)
            materialize(p) = (update!(weights, particles, mesh, othermask); 0)
            @test scatter(viewweights, vp, materialize) ≈ expected
            @test !Tesserae.isdeferred(weights)
        end
    end
end
