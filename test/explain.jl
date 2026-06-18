@testset "Explain macro" begin
    function explain_setup()
        T = Float64
        GridProp = @NamedTuple begin
            x  :: Vec{2,T}
            m  :: T
            mv :: Vec{2,T}
            v  :: Vec{2,T}
            vn :: Vec{2,T}
        end
        ParticleProp = @NamedTuple begin
            x :: Vec{2,T}
            m :: T
            v :: Vec{2,T}
            a :: Vec{2,T}
        end
        mesh = CartesianMesh(1.0, (0,3), (0,2))
        grid = generate_grid(GridProp, mesh)
        particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
        for p in eachindex(particles)
            particles.m[p] = 1 + 0.1p
            particles.v[p] = Vec(sin(p), cos(p))
        end
        for i in eachindex(grid)
            grid.v[i] = Vec(0.1 * first(Tuple(i)), 0.2 * last(Tuple(i)))
            grid.vn[i] = Vec(0.05 * first(Tuple(i)), 0.1 * last(Tuple(i)))
        end
        weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
        update!(weights, particles, mesh)
        grid, particles, weights
    end

    function spgrid_explain_setup()
        T = Float64
        GridProp = @NamedTuple begin
            x  :: Vec{2,T}
            m  :: T
            mv :: Vec{2,T}
            v  :: Vec{2,T}
            vn :: Vec{2,T}
        end
        ParticleProp = @NamedTuple begin
            x :: Vec{2,T}
            m :: T
            v :: Vec{2,T}
            a :: Vec{2,T}
        end

        mesh = CartesianMesh(1.0, (0,31), (0,31); block_size_log2=Val(2))
        grid = generate_grid(SpArray, GridProp, mesh)
        particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
        filter!(particles) do p
            x = p.x
            (x[1] - 2)^2 + (x[2] - 2)^2 < 2
        end
        weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
        update!(weights, particles, mesh)
        update_sparsity!(grid, particles.x)

        for p in eachindex(particles)
            particles.m[p] = 1 + 0.1p
            particles.v[p] = Vec(sin(p), cos(p))
            particles.a[p] = Vec(0.01p, -0.02p)
        end
        for i in Tesserae.activeindices(Tesserae.get_spinds(grid))
            x = grid.x[i]
            grid.v[i] = Vec(0.1 * x[1], 0.2 * x[2])
            grid.vn[i] = Vec(0.05 * x[1], 0.1 * x[2])
        end

        grid, particles, weights
    end

    function explained_function(ex, argnames)
        eval(:(($(argnames...),) -> begin
            $(ex.code)
            nothing
        end))
    end

    explain_text(ex) = sprint(show, MIME("text/plain"), ex)

    function displayed_function(ex, argnames)
        code = Meta.parse("begin\n$(explain_text(ex))\nend")
        eval(:(($(argnames...),) -> begin
            $code
            nothing
        end))
    end

    run_displayed(ex, argnames, args...) = Base.invokelatest(displayed_function(ex, argnames), args...)

    function test_grid_equal(actual, expected)
        @test actual.m ≈ expected.m
        @test actual.mv ≈ expected.mv
        @test actual.vn ≈ expected.vn
        @test actual.v ≈ expected.v
    end

    function test_particles_equal(actual, expected)
        @test actual.x ≈ expected.x
        @test actual.v ≈ expected.v
        @test actual.a ≈ expected.a
    end

    function macroexpand_error(ex)
        try
            macroexpand(@__MODULE__, ex)
        catch err
            return sprint(showerror, err)
        end
        error("expected macro expansion to fail")
    end

    function test_rejected_by_actual_and_explain(actual, explained, needles)
        actual_msg = macroexpand_error(actual)
        explained_msg = macroexpand_error(explained)
        for needle in needles
            @test occursin(needle, actual_msg)
            @test occursin(needle, explained_msg)
        end
    end

    @testset "P2G" begin
        grid, particles, weights = explain_setup()
        grid_ref = deepcopy(grid)
        grid_display = deepcopy(grid)
        grid_macro = deepcopy(grid)
        grid_threaded_ref = deepcopy(grid)
        grid_threaded_display = deepcopy(grid)
        grid_threaded_macro = deepcopy(grid)
        partition = ThreadPartition(Tesserae.get_mesh(grid))
        update!(partition, particles.x)

        ex = @explain @P2G grid=>i particles=>p weights=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        ex_threaded = @explain @threaded @P2G grid=>i particles=>p weights=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        ex_threaded_static = @explain @threaded :static @P2G grid=>i particles=>p weights=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        ex_threaded_partition = @explain @threaded @P2G grid=>i particles=>p weights=>ip partition begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        f = explained_function(ex, (:grid, :particles, :weights))
        f(grid_ref, particles, weights)
        f_display = displayed_function(ex, (:grid, :particles, :weights))
        f_display(grid_display, particles, weights)
        f_threaded = explained_function(ex_threaded_partition, (:grid, :particles, :weights, :partition))
        f_threaded(grid_threaded_ref, particles, weights, partition)
        f_threaded_display = displayed_function(ex_threaded_partition, (:grid, :particles, :weights, :partition))
        f_threaded_display(grid_threaded_display, particles, weights, partition)

        @P2G grid_macro=>i particles=>p weights=>ip begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        @threaded @P2G grid_threaded_macro=>i particles=>p weights=>ip partition begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end

        @test grid_ref.m ≈ grid_macro.m
        @test grid_ref.mv ≈ grid_macro.mv
        @test grid_display.m ≈ grid_macro.m
        @test grid_display.mv ≈ grid_macro.mv
        @test grid_threaded_ref.m ≈ grid_threaded_macro.m
        @test grid_threaded_ref.mv ≈ grid_threaded_macro.mv
        @test grid_threaded_display.m ≈ grid_threaded_macro.m
        @test grid_threaded_display.mv ≈ grid_threaded_macro.mv
        shown = sprint(show, MIME("text/plain"), ex)
        @test occursin("Reference expansion of @P2G", shown)
        @test sprint(show, ex) == shown
        let ExplainedCode = nothing
            @test (@explain @P2G grid=>i particles=>p weights=>ip begin
                m[i] = @∑ w[ip] * m[p]
            end) isa Tesserae.ExplainedCode
        end
        @test !occursin("Tesserae.", shown)
        @test !occursin(r"(?m)^begin\b", shown)
        @test occursin("for p in eachindex(particles)", shown)
        @test !occursin("for p =", shown)
        shown_threaded = sprint(show, MIME("text/plain"), ex_threaded)
        @test occursin("@warn", shown_threaded)
        @test !occursin("Threads.@threads", shown_threaded)
        @test occursin("@warn", sprint(show, MIME("text/plain"), ex_threaded_static))
        @test occursin("@warn", sprint(show, MIME("text/plain"), @explain @P2G :static grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
        end))
        shown_threaded_partition = sprint(show, MIME("text/plain"), ex_threaded_partition)
        @test occursin("for group in threadsafe_groups(partition)", shown_threaded_partition)
        @test occursin("Threads.@threads :dynamic for region in group", shown_threaded_partition)
        @test occursin("for p in particle_indices(partition, particles, region)", shown_threaded_partition)
        @test_throws ErrorException macroexpand(@__MODULE__, quote
            @explain @threaded :static :extra @P2G grid=>i particles=>p weights=>ip begin
                m[i] = @∑ w[ip] * m[p]
            end
        end)
    end

    @testset "G2P" begin
        grid, particles, weights = explain_setup()
        particles_ref = deepcopy(particles)
        particles_display = deepcopy(particles)
        particles_macro = deepcopy(particles)
        particles_threaded_ref = deepcopy(particles)
        particles_threaded_display = deepcopy(particles)
        particles_threaded_macro = deepcopy(particles)
        scale = 0.01

        ex = @explain @G2P grid=>i particles=>p weights=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vn[i])
            a[p]  = @∑ w[ip] * v[i]
            x[p] += a[p] * $scale
        end
        ex_threaded = @explain @threaded @G2P grid=>i particles=>p weights=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vn[i])
            a[p]  = @∑ w[ip] * v[i]
            x[p] += a[p] * $scale
        end
        f = explained_function(ex, (:grid, :particles, :weights, :scale))
        f(grid, particles_ref, weights, scale)
        f_display = displayed_function(ex, (:grid, :particles, :weights, :scale))
        f_display(grid, particles_display, weights, scale)
        f_threaded = explained_function(ex_threaded, (:grid, :particles, :weights, :scale))
        f_threaded(grid, particles_threaded_ref, weights, scale)
        f_threaded_display = displayed_function(ex_threaded, (:grid, :particles, :weights, :scale))
        f_threaded_display(grid, particles_threaded_display, weights, scale)

        @G2P grid=>i particles_macro=>p weights=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vn[i])
            a[p]  = @∑ w[ip] * v[i]
            x[p] += a[p] * $scale
        end
        @threaded @G2P grid=>i particles_threaded_macro=>p weights=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vn[i])
            a[p]  = @∑ w[ip] * v[i]
            x[p] += a[p] * $scale
        end

        @test particles_ref.v ≈ particles_macro.v
        @test particles_ref.a ≈ particles_macro.a
        @test particles_ref.x ≈ particles_macro.x
        @test particles_display.v ≈ particles_macro.v
        @test particles_display.a ≈ particles_macro.a
        @test particles_display.x ≈ particles_macro.x
        @test particles_threaded_ref.v ≈ particles_threaded_macro.v
        @test particles_threaded_ref.a ≈ particles_threaded_macro.a
        @test particles_threaded_ref.x ≈ particles_threaded_macro.x
        @test particles_threaded_display.v ≈ particles_threaded_macro.v
        @test particles_threaded_display.a ≈ particles_threaded_macro.a
        @test particles_threaded_display.x ≈ particles_threaded_macro.x
        @test ex isa ExplainedCode
        @test !occursin("#=", sprint(show, MIME("text/plain"), ex))
        shown_threaded = sprint(show, MIME("text/plain"), ex_threaded)
        @test occursin("Threads.@threads :dynamic for p in eachindex(particles)", shown_threaded)
        @test occursin("\n        i = nodes[ip]", shown_threaded)
        @test !occursin("\ni = nodes[ip]", shown_threaded)
        @test occursin("particles.x[p] += particles.a[p] * scale", shown_threaded)
    end

    @testset "G2P2G" begin
        grid, particles, weights = explain_setup()
        grid_ref = deepcopy(grid)
        grid_display = deepcopy(grid)
        grid_macro = deepcopy(grid)
        grid_threaded_ref = deepcopy(grid)
        grid_threaded_display = deepcopy(grid)
        grid_threaded_macro = deepcopy(grid)
        particles_ref = deepcopy(particles)
        particles_display = deepcopy(particles)
        particles_macro = deepcopy(particles)
        particles_threaded_ref = deepcopy(particles)
        particles_threaded_display = deepcopy(particles)
        particles_threaded_macro = deepcopy(particles)
        partition = ThreadPartition(Tesserae.get_mesh(grid))
        update!(partition, particles.x)

        ex = @explain @G2P2G grid=>i particles=>p weights=>ip begin
            a[p] = @∑ w[ip] * v[i]
            v[p] += a[p] * 0.1
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        ex_threaded = @explain @threaded @G2P2G grid=>i particles=>p weights=>ip partition begin
            a[p] = @∑ w[ip] * v[i]
            v[p] += a[p] * 0.1
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        f = explained_function(ex, (:grid, :particles, :weights))
        f(grid_ref, particles_ref, weights)
        f_display = displayed_function(ex, (:grid, :particles, :weights))
        f_display(grid_display, particles_display, weights)
        f_threaded = explained_function(ex_threaded, (:grid, :particles, :weights, :partition))
        f_threaded(grid_threaded_ref, particles_threaded_ref, weights, partition)
        f_threaded_display = displayed_function(ex_threaded, (:grid, :particles, :weights, :partition))
        f_threaded_display(grid_threaded_display, particles_threaded_display, weights, partition)

        @G2P2G grid_macro=>i particles_macro=>p weights=>ip begin
            a[p] = @∑ w[ip] * v[i]
            v[p] += a[p] * 0.1
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        @threaded @G2P2G grid_threaded_macro=>i particles_threaded_macro=>p weights=>ip partition begin
            a[p] = @∑ w[ip] * v[i]
            v[p] += a[p] * 0.1
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end

        @test particles_ref.a ≈ particles_macro.a
        @test particles_ref.v ≈ particles_macro.v
        @test grid_ref.m ≈ grid_macro.m
        @test grid_ref.mv ≈ grid_macro.mv
        @test particles_display.a ≈ particles_macro.a
        @test particles_display.v ≈ particles_macro.v
        @test grid_display.m ≈ grid_macro.m
        @test grid_display.mv ≈ grid_macro.mv
        @test particles_threaded_ref.a ≈ particles_threaded_macro.a
        @test particles_threaded_ref.v ≈ particles_threaded_macro.v
        @test grid_threaded_ref.m ≈ grid_threaded_macro.m
        @test grid_threaded_ref.mv ≈ grid_threaded_macro.mv
        @test particles_threaded_display.a ≈ particles_threaded_macro.a
        @test particles_threaded_display.v ≈ particles_threaded_macro.v
        @test grid_threaded_display.m ≈ grid_threaded_macro.m
        @test grid_threaded_display.mv ≈ grid_threaded_macro.mv
        shown = sprint(show, MIME("text/plain"), ex)
        @test count(_ -> true, eachmatch(r"bw = weights\[p\]", shown)) == 1
        @test count(_ -> true, eachmatch(r"nodes = supportnodes\(bw, grid\)", shown)) == 1
        @test occursin("Threads.@threads :dynamic for region in group", sprint(show, MIME("text/plain"), ex_threaded))
    end

    @testset "Invalid transfer programs" begin
        test_rejected_by_actual_and_explain(
            quote
                @P2G grid=>i particles=>p weights=>ip begin
                    a[p] = @∑ w[ip] * v[p]
                end
            end,
            quote
                @explain @P2G grid=>i particles=>p weights=>ip begin
                    a[p] = @∑ w[ip] * v[p]
                end
            end,
            ("invalid LHS index", "must be [i]"),
        )

        test_rejected_by_actual_and_explain(
            quote
                @G2P grid=>i particles=>p weights=>ip begin
                    v_pic = @∑ w[ip] * v[i]
                end
            end,
            quote
                @explain @G2P grid=>i particles=>p weights=>ip begin
                    v_pic = @∑ w[ip] * v[i]
                end
            end,
            ("invalid LHS", "v_pic"),
        )

        test_rejected_by_actual_and_explain(
            quote
                @G2P grid=>i particles=>p weights=>ip begin
                    v[p] += a[p]
                    a[p] = @∑ w[ip] * v[i]
                end
            end,
            quote
                @explain @G2P grid=>i particles=>p weights=>ip begin
                    v[p] += a[p]
                    a[p] = @∑ w[ip] * v[i]
                end
            end,
            ("Equations without `@∑`", "must come after"),
        )

        test_rejected_by_actual_and_explain(
            quote
                @G2P2G grid=>i particles=>p weights=>ip begin
                    m[i] = @∑ w[ip] * m[p]
                    a[p] = @∑ w[ip] * v[i]
                end
            end,
            quote
                @explain @G2P2G grid=>i particles=>p weights=>ip begin
                    m[i] = @∑ w[ip] * m[p]
                    a[p] = @∑ w[ip] * v[i]
                end
            end,
            ("particle `@∑` equations", "must come before"),
        )

        test_rejected_by_actual_and_explain(
            quote
                @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                    A[i,j] = 1
                end
            end,
            quote
                @explain @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                    A[i,j] = 1
                end
            end,
            ("P2G_Matrix", "all equations must use `@∑`"),
        )

        test_rejected_by_actual_and_explain(
            quote
                @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                    A[i,k] = @∑ w[ip] * w[jp]
                end
            end,
            quote
                @explain @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
                    A[i,k] = @∑ w[ip] * w[jp]
                end
            end,
            ("P2G_Matrix", "A[i, k]"),
        )
    end

    @testset "SpGrid" begin
        Δt = 0.01
        gravity = Vec(0.0, -9.81)

        grid, particles, weights = spgrid_explain_setup()
        @test length(collect(Tesserae.activeindices(Tesserae.get_spinds(grid)))) < prod(size(Tesserae.get_mesh(grid)))
        @test eltype(supportnodes(weights[1], grid)) <: Tesserae.SpIndex

        grid_display = deepcopy(grid)
        grid_macro = deepcopy(grid)
        ex = @explain @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            invm = iszero(m[i]) ? zero(m[i]) : inv(m[i])
            vn[i] = mv[i] * invm
            v[i] = vn[i] + gravity * Δt
        end
        run_displayed(ex, (:grid, :particles, :weights, :gravity, :Δt), grid_display, particles, weights, gravity, Δt)
        @P2G grid_macro=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            invm = iszero(m[i]) ? zero(m[i]) : inv(m[i])
            vn[i] = mv[i] * invm
            v[i] = vn[i] + gravity * Δt
        end
        test_grid_equal(grid_display, grid_macro)

        grid, particles, weights = spgrid_explain_setup()
        partition = ThreadPartition(Tesserae.get_mesh(grid))
        update!(partition, particles.x)
        grid_threaded_display = deepcopy(grid)
        grid_threaded_macro = deepcopy(grid)
        ex_threaded = @explain @threaded @P2G grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            invm = iszero(m[i]) ? zero(m[i]) : inv(m[i])
            vn[i] = mv[i] * invm
            v[i] = vn[i] + gravity * Δt
        end
        run_displayed(ex_threaded, (:grid, :particles, :weights, :partition, :gravity, :Δt), grid_threaded_display, particles, weights, partition, gravity, Δt)
        @threaded @P2G grid_threaded_macro=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            invm = iszero(m[i]) ? zero(m[i]) : inv(m[i])
            vn[i] = mv[i] * invm
            v[i] = vn[i] + gravity * Δt
        end
        test_grid_equal(grid_threaded_display, grid_threaded_macro)

        grid, particles, weights = spgrid_explain_setup()
        particles_display = deepcopy(particles)
        particles_macro = deepcopy(particles)
        ex = @explain @G2P grid=>i particles=>p weights=>ip begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            x[p] += @∑ w[ip] * v[i] * Δt
            v[p] += a[p] * Δt
        end
        run_displayed(ex, (:grid, :particles, :weights, :Δt), grid, particles_display, weights, Δt)
        @G2P grid=>i particles_macro=>p weights=>ip begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            x[p] += @∑ w[ip] * v[i] * Δt
            v[p] += a[p] * Δt
        end
        test_particles_equal(particles_display, particles_macro)

        grid, particles, weights = spgrid_explain_setup()
        particles_threaded_display = deepcopy(particles)
        particles_threaded_macro = deepcopy(particles)
        ex_threaded = @explain @threaded @G2P grid=>i particles=>p weights=>ip begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            x[p] += @∑ w[ip] * v[i] * Δt
            v[p] += a[p] * Δt
        end
        run_displayed(ex_threaded, (:grid, :particles, :weights, :Δt), grid, particles_threaded_display, weights, Δt)
        @threaded @G2P grid=>i particles_threaded_macro=>p weights=>ip begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            x[p] += @∑ w[ip] * v[i] * Δt
            v[p] += a[p] * Δt
        end
        test_particles_equal(particles_threaded_display, particles_threaded_macro)

        grid, particles, weights = spgrid_explain_setup()
        grid_display = deepcopy(grid)
        grid_macro = deepcopy(grid)
        particles_display = deepcopy(particles)
        particles_macro = deepcopy(particles)
        ex = @explain @G2P2G grid=>i particles=>p weights=>ip begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            v[p] += a[p] * Δt
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        run_displayed(ex, (:grid, :particles, :weights, :Δt), grid_display, particles_display, weights, Δt)
        @G2P2G grid_macro=>i particles_macro=>p weights=>ip begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            v[p] += a[p] * Δt
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        test_grid_equal(grid_display, grid_macro)
        test_particles_equal(particles_display, particles_macro)

        grid, particles, weights = spgrid_explain_setup()
        partition = ThreadPartition(Tesserae.get_mesh(grid))
        update!(partition, particles.x)
        grid_threaded_display = deepcopy(grid)
        grid_threaded_macro = deepcopy(grid)
        particles_threaded_display = deepcopy(particles)
        particles_threaded_macro = deepcopy(particles)
        ex_threaded = @explain @threaded @G2P2G grid=>i particles=>p weights=>ip partition begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            v[p] += a[p] * Δt
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        run_displayed(ex_threaded, (:grid, :particles, :weights, :partition, :Δt), grid_threaded_display, particles_threaded_display, weights, partition, Δt)
        @threaded @G2P2G grid_threaded_macro=>i particles_threaded_macro=>p weights=>ip partition begin
            a[p] = @∑ w[ip] * ((v[i] - vn[i]) / Δt)
            v[p] += a[p] * Δt
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
        end
        test_grid_equal(grid_threaded_display, grid_threaded_macro)
        test_particles_equal(particles_threaded_display, particles_threaded_macro)

        grid, particles, weights = spgrid_explain_setup()
        mesh = Tesserae.get_mesh(grid)
        basis = BSpline(Linear())
        A_display = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        A_macro = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        ex = @explain @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        run_displayed(ex, (:A, :grid, :particles, :weights), A_display, grid, particles, weights)
        A = A_macro
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        @test A_display ≈ A_macro

        grid, particles, weights = spgrid_explain_setup()
        partition = ThreadPartition(Tesserae.get_mesh(grid))
        update!(partition, particles.x)
        A_threaded_display = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        A_threaded_macro = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        ex_threaded = @explain @threaded @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        run_displayed(ex_threaded, (:A, :grid, :particles, :weights, :partition), A_threaded_display, grid, particles, weights, partition)
        A = A_threaded_macro
        @threaded @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        @test A_threaded_display ≈ A_threaded_macro
    end

    @testset "P2G_Matrix" begin
        basis = BSpline(Quadratic())
        mesh = CartesianMesh(1, (0,4), (0,5))
        grid = generate_grid(@NamedTuple{x::Vec{2,Float64}}, mesh)
        particles = generate_particles(@NamedTuple{x::Vec{2,Float64}}, mesh; alg=GridSampling())
        weights = generate_basis_weights(basis, mesh, length(particles))
        update!(weights, particles, mesh)

        A_ref = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        A_display = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        A_macro = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        A_threaded_ref = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        A_threaded_display = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        A_threaded_macro = create_sparse_matrix(basis, mesh; ndofs=(2, 2))
        partition = ThreadPartition(mesh)
        update!(partition, particles.x)

        ex = @explain @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        ex_threaded = @explain @threaded @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        f = explained_function(ex, (:A, :grid, :particles, :weights))
        f(A_ref, grid, particles, weights)
        f_display = displayed_function(ex, (:A, :grid, :particles, :weights))
        f_display(A_display, grid, particles, weights)
        f_threaded = explained_function(ex_threaded, (:A, :grid, :particles, :weights, :partition))
        f_threaded(A_threaded_ref, grid, particles, weights, partition)
        f_threaded_display = displayed_function(ex_threaded, (:A, :grid, :particles, :weights, :partition))
        f_threaded_display(A_threaded_display, grid, particles, weights, partition)

        A = A_macro
        @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end
        A = A_threaded_macro
        @threaded @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) partition begin
            A[i,j] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end

        @test A_ref ≈ A_macro
        @test A_display ≈ A_macro
        @test A_threaded_ref ≈ A_threaded_macro
        @test A_threaded_display ≈ A_threaded_macro
        shown = sprint(show, MIME("text/plain"), ex)
        @test occursin("Tesserae.matrix_dof_tables(A, grid, grid)", shown)
        @test occursin("Tesserae.matrix_supportnodes(bw_i, grid)", shown)
        @test occursin("Tesserae.local_dof_table(A_table_i, nodes_i)", shown)
        @test occursin("Tesserae.local_dofs(A_local_i, ip)", shown)
        @test occursin("Tesserae.support_dofs(A_table_i, nodes_i, A_table_j, nodes_j)", shown)
        @test occursin("A[A_dofs_i, A_dofs_j] .+= A_local", shown)
        @test occursin("A_local[A_i, A_j] .= ", shown)
        @test !occursin("A_lmat", shown)
        @test !occursin("A_I", shown)
        @test !occursin("A_J", shown)
        @test !occursin("Tesserae.add!", shown)
        @test !occursin("LinearIndices", shown)
        @test !occursin("vec(view", shown)
        @test !occursin(r"(?m)^begin\b", shown)
        @test occursin("for ip in eachindex(nodes_i)", shown)
        @test !occursin("for ip =", shown)
        @test occursin("Threads.@threads :dynamic for region in group", sprint(show, MIME("text/plain"), ex_threaded))

        shown_transposed = sprint(show, MIME("text/plain"), @explain @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[j,i] = @∑ ∇w[ip] ⊗ ∇w[jp]
        end)
        @test occursin("(A_table_j, A_table_i) = Tesserae.matrix_dof_tables(A, grid, grid)", shown_transposed)
        @test occursin("A[A_dofs_j, A_dofs_i] .+= A_local'", shown_transposed)
    end

    @testset "display fixtures" begin
        @test explain_text(@explain @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ begin
                contrib = w[ip] * m[p]
                contrib
            end
            v[i] += begin
                dv = f[i] * dt
                dv
            end
        end) == """
# Reference expansion of @P2G.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

fillzero!(grid.m)
for p in eachindex(particles)
    bw = weights[p]
    nodes = supportnodes(bw, grid)
    for ip in eachindex(nodes)
        i = nodes[ip]
        grid.m[i] += begin
            contrib = bw.w[ip] * particles.m[p]
            contrib
        end
    end
end
for i in eachindex(grid)
    grid.v[i] += begin
        dv = grid.f[i] * dt
        dv
    end
end"""

        @test explain_text(@explain @P2G grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i] = @∑ -V⁰[p] * det(F[p]) * σ[p] * ∇w[ip]

            mᵢ⁻¹ = iszero(m[i]) ? zero(m[i]) : inv(m[i])
            vⁿ[i] = mv[i] * mᵢ⁻¹
            v[i] = vⁿ[i] + (f[i] * mᵢ⁻¹) * Δt
        end) == """
# Reference expansion of @P2G.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

fillzero!(grid.m)
fillzero!(grid.mv)
fillzero!(grid.f)
for p in eachindex(particles)
    bw = weights[p]
    nodes = supportnodes(bw, grid)
    for ip in eachindex(nodes)
        i = nodes[ip]
        grid.m[i] += bw.w[ip] * particles.m[p]
        grid.mv[i] += bw.w[ip] * particles.m[p] * particles.v[p]
        grid.f[i] += -(particles.V⁰[p]) * det(particles.F[p]) * particles.σ[p] * bw.∇w[ip]
    end
end
for i in eachindex(grid)
    mᵢ⁻¹ = if iszero(grid.m[i])
        zero(grid.m[i])
    else
        inv(grid.m[i])
    end
    grid.vⁿ[i] = grid.mv[i] * mᵢ⁻¹
    grid.v[i] = grid.vⁿ[i] + (grid.f[i] * mᵢ⁻¹) * Δt
end"""

        @test explain_text(@explain @threaded @P2G grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p]
        end) == """
# Reference expansion of @P2G.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

fillzero!(grid.m)
for group in threadsafe_groups(partition)
    Threads.@threads :dynamic for region in group
        for p in particle_indices(partition, particles, region)
            bw = weights[p]
            nodes = supportnodes(bw, grid)
            for ip in eachindex(nodes)
                i = nodes[ip]
                grid.m[i] += bw.w[ip] * particles.m[p]
            end
        end
    end
end"""

        @test explain_text(@explain @G2P grid=>i particles=>p weights=>ip begin
            v[p] += @∑ begin
                dv = w[ip] * (v[i] - vn[i])
                dv
            end
            a[p] = @∑ begin
                force = w[ip] * f[i]
                force
            end
            x[p] += begin
                dx = v[p] * dt
                dx
            end
        end) == """
# Reference expansion of @G2P.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

for p in eachindex(particles)
    v_sum = zero(eltype(particles.v))
    a_sum = zero(eltype(particles.a))
    bw = weights[p]
    nodes = supportnodes(bw, grid)
    for ip in eachindex(nodes)
        i = nodes[ip]
        v_sum += begin
            dv = bw.w[ip] * (grid.v[i] - grid.vn[i])
            dv
        end
        a_sum += begin
            force = bw.w[ip] * grid.f[i]
            force
        end
    end
    particles.v[p] += v_sum
    particles.a[p] = a_sum
    particles.x[p] += begin
        dx = particles.v[p] * dt
        dx
    end
end"""

        @test explain_text(@explain @G2P grid=>i particles=>p weights=>ip begin
            v[p] += @∑ w[ip] * (v[i] - vⁿ[i])
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt

            Δεₚ = symmetric(∇v[p]) * Δt
            F[p] = (I + ∇v[p] * Δt) * F[p]
            σ[p] += λ * tr(Δεₚ) * I + 2μ * Δεₚ
        end) == """
# Reference expansion of @G2P.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

for p in eachindex(particles)
    v_sum = zero(eltype(particles.v))
    ∇v_sum = zero(eltype(particles.∇v))
    x_sum = zero(eltype(particles.x))
    bw = weights[p]
    nodes = supportnodes(bw, grid)
    for ip in eachindex(nodes)
        i = nodes[ip]
        v_sum += bw.w[ip] * (grid.v[i] - grid.vⁿ[i])
        ∇v_sum += grid.v[i] ⊗ bw.∇w[ip]
        x_sum += bw.w[ip] * grid.v[i] * Δt
    end
    particles.v[p] += v_sum
    particles.∇v[p] = ∇v_sum
    particles.x[p] += x_sum
    Δεₚ = symmetric(particles.∇v[p]) * Δt
    particles.F[p] = (I + particles.∇v[p] * Δt) * particles.F[p]
    particles.σ[p] += λ * tr(Δεₚ) * I + (2μ) * Δεₚ
end"""

        @test explain_text(@explain @G2P2G grid=>i particles=>p weights=>ip begin
            a[p] = @∑ begin
                ap = w[ip] * v[i]
                ap
            end
            v[p] += begin
                dv = a[p] * dt
                dv
            end
            m[i] = @∑ begin
                mi = w[ip] * m[p]
                mi
            end
        end) == """
# Reference expansion of @G2P2G.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

fillzero!(grid.m)
for p in eachindex(particles)
    a_sum = zero(eltype(particles.a))
    bw = weights[p]
    nodes = supportnodes(bw, grid)
    for ip in eachindex(nodes)
        i = nodes[ip]
        a_sum += begin
            ap = bw.w[ip] * grid.v[i]
            ap
        end
    end
    particles.a[p] = a_sum
    particles.v[p] += begin
        dv = particles.a[p] * dt
        dv
    end
    for ip in eachindex(nodes)
        i = nodes[ip]
        grid.m[i] += begin
            mi = bw.w[ip] * particles.m[p]
            mi
        end
    end
end"""

        @test explain_text(@explain @P2G_Matrix grid=>(i,j) particles=>p weights=>(ip,jp) begin
            A[i,j] = @∑ begin
                K = ∇w[ip] ⊗ ∇w[jp]
                K
            end
        end) == """
# Reference expansion of @P2G_Matrix.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

fillzero!(A)
(A_table_i, A_table_j) = Tesserae.matrix_dof_tables(A, grid, grid)
for p in eachindex(particles)
    bw_i = weights[p]
    bw_j = bw_i
    (nodes_i, nodes_j) = Tesserae.matrix_supportnodes(bw_i, grid)
    A_local_i = Tesserae.local_dof_table(A_table_i, nodes_i)
    A_local_j = Tesserae.local_dof_table(A_table_j, nodes_j)
    A_local = zeros(eltype(A), (length(A_local_i), length(A_local_j)))
    for jp in eachindex(nodes_j)
        j = nodes_j[jp]
        A_j = Tesserae.local_dofs(A_local_j, jp)
        for ip in eachindex(nodes_i)
            i = nodes_i[ip]
            A_i = Tesserae.local_dofs(A_local_i, ip)
            A_local[A_i, A_j] .= begin
                K = bw_i.∇w[ip] ⊗ bw_j.∇w[jp]
                K
            end
        end
    end
    (A_dofs_i, A_dofs_j) = Tesserae.support_dofs(A_table_i, nodes_i, A_table_j, nodes_j)
    A[A_dofs_i, A_dofs_j] .+= A_local
end"""

        @test explain_text(@explain @P2G_Matrix (grid_u,grid_p)=>(i,j) particles=>p (weights_u,weights_p)=>(ip,jp) begin
            A[j,i] = @∑ ∇N[ip] * N[jp]
        end) == """
# Reference expansion of @P2G_Matrix.
# Runnable CPU code for understanding/debugging.
# This is not the optimized lowering used by the macro.

fillzero!(A)
(A_table_j, A_table_i) = Tesserae.matrix_dof_tables(A, grid_p, grid_u)
for p in eachindex(particles)
    bw_i = weights_u[p]
    bw_j = weights_p[p]
    (nodes_i, nodes_j) = Tesserae.matrix_supportnodes(bw_i, grid_u, bw_j, grid_p)
    A_local_i = Tesserae.local_dof_table(A_table_i, nodes_i)
    A_local_j = Tesserae.local_dof_table(A_table_j, nodes_j)
    A_local = zeros(eltype(A), (length(A_local_i), length(A_local_j)))
    for jp in eachindex(nodes_j)
        j = nodes_j[jp]
        A_j = Tesserae.local_dofs(A_local_j, jp)
        for ip in eachindex(nodes_i)
            i = nodes_i[ip]
            A_i = Tesserae.local_dofs(A_local_i, ip)
            A_local[A_i, A_j] .= bw_i.∇N[ip] * bw_j.N[jp]
        end
    end
    (A_dofs_i, A_dofs_j) = Tesserae.support_dofs(A_table_i, nodes_i, A_table_j, nodes_j)
    A[A_dofs_j, A_dofs_i] .+= A_local'
end"""
    end
end
