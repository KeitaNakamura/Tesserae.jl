@testset "Transfer macros" begin
    function transfer_fixture()
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        GridProp = @NamedTuple begin
            x   :: Vec{2, Float64}
            m   :: Float64
            m⁻¹ :: Float64
            mv  :: Vec{2, Float64}
            f   :: Vec{2, Float64}
            v   :: Vec{2, Float64}
            vⁿ  :: Vec{2, Float64}
        end
        ParticleProp = @NamedTuple begin
            x  :: Vec{2, Float64}
            m  :: Float64
            V  :: Float64
            v  :: Vec{2, Float64}
            ∇v :: SecondOrderTensor{2, Float64, 4}
            F  :: SecondOrderTensor{2, Float64, 4}
            σ  :: SecondOrderTensor{2, Float64, 4}
        end

        grid = generate_grid(GridProp, mesh)
        particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
        weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
        update!(weights, particles, mesh)
        initialize_mpm_state!(grid, particles)
        grid, particles, weights
    end

    function cpdi_spgrid_fixture()
        mesh = CartesianMesh(1.0, (0,2), (0,2))
        GridProp = @NamedTuple begin
            x :: Vec{2, Float64}
            m :: Float64
            v :: Vec{2, Float64}
        end
        ParticleProp = @NamedTuple begin
            x :: Vec{2, Float64}
            m :: Float64
            v :: Vec{2, Float64}
            F :: SecondOrderTensor{2, Float64, 4}
            l :: Float64
        end

        grid = generate_grid(SpArray, GridProp, mesh)
        particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
        weights = generate_basis_weights(CPDI(), mesh, length(particles))
        grid, particles, weights
    end

    function error_message(f)
        try
            f()
            nothing
        catch err
            sprint(showerror, err)
        end
    end

    function initialize_mpm_state!(grid, particles)
        for p in eachindex(particles)
            particles.m[p] = 1.0 + 0.05p
            particles.V[p] = 0.2 + 0.01p
            particles.v[p] = Vec(0.15 + 0.02p, -0.25 + 0.015p)
            particles.∇v[p] = zero(eltype(particles.∇v))
            particles.F[p] = diagm(Vec(1.0 + 0.002p, 1.0 - 0.001p))
            particles.σ[p] = symmetric(Vec(1.0 + 0.03p, 0.2) ⊗ Vec(0.4, 0.8 + 0.01p))
        end

        for i in eachindex(grid)
            I = Tuple(i)
            grid.m[i] = 1.0
            grid.m⁻¹[i] = 1.0
            grid.mv[i] = Vec(0.0, 0.0)
            grid.f[i] = Vec(0.0, 0.0)
            grid.vⁿ[i] = Vec(0.08 * I[1], -0.04 * I[2])
            grid.v[i] = grid.vⁿ[i] + Vec(0.03 * I[2], -0.02 * I[1])
        end

        nothing
    end

    function manual_p2g!(grid, particles, weights, Δt, gravity)
        fillzero!(grid.m)
        fillzero!(grid.mv)
        fillzero!(grid.f)

        for p in eachindex(particles)
            bw = weights[p]
            nodeindices = supportnodes(bw, grid)
            for ip in eachindex(nodeindices)
                i = nodeindices[ip]
                grid.m[i] += bw.w[ip] * particles.m[p]
                grid.mv[i] += bw.w[ip] * particles.m[p] * particles.v[p]
                grid.f[i] += bw.w[ip] * particles.m[p] * gravity
                grid.f[i] -= particles.V[p] * particles.σ[p] * bw.∇w[ip]
            end
        end

        for i in eachindex(grid)
            grid.m⁻¹[i] = inv(grid.m[i]) * !iszero(grid.m[i])
            grid.vⁿ[i] = grid.mv[i] * grid.m⁻¹[i]
            grid.v[i] = grid.vⁿ[i] + (grid.f[i] * grid.m⁻¹[i]) * Δt
        end

        grid
    end

    function manual_g2p_pic_flip!(grid, particles, weights, α, Δt)
        for p in eachindex(particles)
            v_pic = zero(eltype(particles.v))
            Δv_flip = zero(eltype(particles.v))
            ∇v = zero(eltype(particles.∇v))
            Δx = zero(eltype(particles.x))
            bw = weights[p]
            nodeindices = supportnodes(bw, grid)
            for ip in eachindex(nodeindices)
                i = nodeindices[ip]
                v_pic += bw.w[ip] * grid.v[i]
                Δv_flip += bw.w[ip] * (grid.v[i] - grid.vⁿ[i])
                ∇v += grid.v[i] ⊗ bw.∇w[ip]
                Δx += bw.w[ip] * grid.v[i] * Δt
            end
            particles.v[p] = (1 - α) * v_pic + α * (particles.v[p] + Δv_flip)
            particles.∇v[p] = ∇v
            particles.x[p] += Δx
            particles.F[p] = (one(particles.F[p]) + particles.∇v[p] * Δt) * particles.F[p]
        end

        particles
    end

    function manual_g2p2g_internal_force!(grid, particles, weights, Δt, stiffness)
        for p in eachindex(particles)
            ∇v = zero(eltype(particles.∇v))
            bw = weights[p]
            nodeindices = supportnodes(bw, grid)
            for ip in eachindex(nodeindices)
                i = nodeindices[ip]
                ∇v += grid.v[i] ⊗ bw.∇w[ip]
            end
            particles.∇v[p] = ∇v
            particles.F[p] = (one(particles.F[p]) + particles.∇v[p] * Δt) * particles.F[p]
            particles.σ[p] = stiffness * symmetric(particles.∇v[p])
        end

        fillzero!(grid.f)
        for p in eachindex(particles)
            bw = weights[p]
            nodeindices = supportnodes(bw, grid)
            for ip in eachindex(nodeindices)
                i = nodeindices[ip]
                grid.f[i] -= particles.V[p] * particles.σ[p] * bw.∇w[ip]
            end
        end

        for i in eachindex(grid)
            grid.v[i] = grid.vⁿ[i] + (grid.f[i] * grid.m⁻¹[i]) * Δt
        end

        grid, particles
    end

    @testset "@P2G" begin
        Δt = 0.01
        gravity = Vec(0.0, -9.81)
        grid, particles, weights = transfer_fixture()
        expected = deepcopy(grid)
        actual = deepcopy(grid)

        manual_p2g!(expected, particles, weights, Δt, gravity)

        @P2G actual=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i] = @∑ w[ip] * m[p] * gravity
            f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
            invm = inv(m[i]) * !iszero(m[i])
            m⁻¹[i] = invm
            vⁿ[i] = mv[i] * invm
            v[i] = vⁿ[i] + (f[i] * m⁻¹[i]) * Δt
        end

        @test actual.m ≈ expected.m
        @test actual.m⁻¹ ≈ expected.m⁻¹
        @test actual.mv ≈ expected.mv
        @test actual.f ≈ expected.f
        @test actual.vⁿ ≈ expected.vⁿ
        @test actual.v ≈ expected.v
    end

    @testset "transfer with views" begin
        grid, particles, weights = transfer_fixture()
        expected = deepcopy(grid)
        actual = deepcopy(grid)
        subset = 2:length(particles)-1
        particle_view = view(particles, subset)
        weight_view = view(weights, subset)

        fillzero!(expected.m)
        for p in eachindex(particle_view)
            bw = weight_view[p]
            for (ip, i) in enumerate(supportnodes(bw, expected))
                expected.m[i] += bw.w[ip] * particle_view.m[p]
            end
        end

        @P2G actual=>i particle_view=>p weight_view=>ip begin
            m[i] = @∑ w[ip] * m[p]
        end

        @test actual.m ≈ expected.m
    end

    @testset "P2G RHS product hoisting" begin
        hoist_exprs = Any[]
        rhs = Tesserae.hoist_p2g_rhs!(hoist_exprs, Set([:wi, :∇wi]), :(2 * a * b * wi * c * d * ∇wi * e * f))
        hoisted_symbols = map(ex -> ex.args[1], hoist_exprs)

        @test map(ex -> ex.args[2], hoist_exprs) == [:(2 * a * b), :(c * d), :(e * f)]
        @test rhs == Expr(:call, :*, hoisted_symbols[1], :wi, hoisted_symbols[2], :∇wi, hoisted_symbols[3])
    end

    @testset "CPDI rejects SpGrid" begin
        grid, particles, weights = cpdi_spgrid_fixture()

        p2g_err = error_message() do
            @P2G grid=>i particles=>p weights=>ip begin
                m[i] = @∑ w[ip] * m[p]
            end
        end
        @test p2g_err isa String && occursin("@P2G: CPDI is currently supported only on dense Grid, not SpGrid", p2g_err)

        g2p_err = error_message() do
            @G2P grid=>i particles=>p weights=>ip begin
                v[p] = @∑ w[ip] * v[i]
            end
        end
        @test g2p_err isa String && occursin("@G2P: CPDI is currently supported only on dense Grid, not SpGrid", g2p_err)
    end

    @testset "@G2P" begin
        α = 0.95
        Δt = 0.01
        gravity = Vec(0.0, -9.81)
        grid, particles, weights = transfer_fixture()
        manual_p2g!(grid, particles, weights, Δt, gravity)
        expected = deepcopy(particles)
        actual = deepcopy(particles)

        manual_g2p_pic_flip!(grid, expected, weights, α, Δt)

        @G2P grid=>i actual=>p weights=>ip begin
            v[p] = @∑ w[ip] * ((1 - α) * v[i] + α * (v[p] + (v[i] - vⁿ[i])))
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt
            F[p] = (one(F[p]) + ∇v[p] * Δt) * F[p]
        end

        @test actual.v ≈ expected.v
        @test actual.∇v ≈ expected.∇v
        @test actual.x ≈ expected.x
        @test actual.F ≈ expected.F
    end

    @testset "@G2P2G" begin
        Δt = 0.01
        stiffness = 2.5
        gravity = Vec(0.0, -9.81)
        grid, particles, weights = transfer_fixture()
        manual_p2g!(grid, particles, weights, Δt, gravity)
        expected_grid = deepcopy(grid)
        expected_particles = deepcopy(particles)
        actual_grid = deepcopy(grid)
        actual_particles = deepcopy(particles)

        manual_g2p2g_internal_force!(expected_grid, expected_particles, weights, Δt, stiffness)

        @G2P2G actual_grid=>i actual_particles=>p weights=>ip begin
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            F[p] = (one(F[p]) + ∇v[p] * Δt) * F[p]
            σ[p] = stiffness * symmetric(∇v[p])
            f[i] = @∑ -V[p] * σ[p] * ∇w[ip]
            Δv = (f[i] * m⁻¹[i]) * Δt
            v[i] = vⁿ[i] + Δv
        end

        @test actual_particles.∇v ≈ expected_particles.∇v
        @test actual_particles.F ≈ expected_particles.F
        @test actual_particles.σ ≈ expected_particles.σ
        @test actual_grid.f ≈ expected_grid.f
        @test actual_grid.v ≈ expected_grid.v

        expanded = sprint(show, MIME("text/plain"), macroexpand(@__MODULE__, quote
            @G2P2G grid=>i particles=>p weights=>ip begin
                a[p] = @∑ w[ip] * v[i]
                v[p] += a[p] * Δt
                m[i] = @∑ w[ip] * m[p]
            end
        end))
        @test count(_ -> true, eachmatch(r"transfer_support_window\(", expanded)) == 1
        @test occursin("Tesserae.G2P2G", expanded)

        # Without particle `@∑` equations the G2P half binds no support window,
        # so the P2G half must bind it itself.
        expanded = sprint(show, MIME("text/plain"), macroexpand(@__MODULE__, quote
            @G2P2G grid=>i particles=>p weights=>ip begin
                v[p] += a[p] * Δt
                m[i] = @∑ w[ip] * m[p]
            end
        end))
        @test count(_ -> true, eachmatch(r"transfer_support_window\(", expanded)) == 1
    end

    @testset "interpolation" begin
        grid, particles, weights = transfer_fixture()

        p2g_scale = 2.0
        p2g_captures = Ref(0)
        expected_grid = deepcopy(grid)
        actual_grid = deepcopy(grid)

        @P2G expected_grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p] * p2g_scale
            v[i] = x[i] * (p2g_scale + 1)
        end

        @P2G actual_grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p] * $(begin
                p2g_captures[] += 1
                p2g_scale
            end)
            v[i] = x[i] * $(begin
                p2g_captures[] += 1
                p2g_scale + 1
            end)
        end

        @test p2g_captures[] == 2
        @test actual_grid.m ≈ expected_grid.m
        @test actual_grid.v ≈ expected_grid.v

        g2p_scale = 3.0
        g2p_captures = Ref(0)
        expected_particles = deepcopy(particles)
        actual_particles = deepcopy(particles)

        @G2P grid=>i expected_particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * v[i] * g2p_scale
        end

        @G2P grid=>i actual_particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * v[i] * $(begin
                g2p_captures[] += 1
                g2p_scale
            end)
        end

        @test g2p_captures[] == 1
        @test actual_particles.v ≈ expected_particles.v

        g2p2g_scale = 4.0
        g2p2g_captures = Ref(0)
        expected_grid = deepcopy(grid)
        expected_particles = deepcopy(particles)
        actual_grid = deepcopy(grid)
        actual_particles = deepcopy(particles)

        @G2P2G expected_grid=>i expected_particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * v[i] * g2p2g_scale
            m[i] = @∑ w[ip] * m[p] * g2p2g_scale
        end

        @G2P2G actual_grid=>i actual_particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * v[i] * $(begin
                g2p2g_captures[] += 1
                g2p2g_scale
            end)
            m[i] = @∑ w[ip] * m[p] * $(begin
                g2p2g_captures[] += 1
                g2p2g_scale
            end)
        end

        @test g2p2g_captures[] == 2
        @test actual_particles.v ≈ expected_particles.v
        @test actual_grid.m ≈ expected_grid.m

        ex = Meta.parse(raw"""
            @P2G grid=>i particles=>p weights=>ip begin
                $m[i] = @∑ w[ip] * m[p]
            end
        """)
        @test_throws ErrorException macroexpand(@__MODULE__, ex)
    end

    @testset "threaded matches sequential" begin
        Δt = 0.01
        gravity = Vec(0.0, -9.81)
        grid, particles, weights = transfer_fixture()
        partition = ThreadPartition(grid.x)
        update!(partition, particles.x)

        sequential_grid = deepcopy(grid)
        threaded_grid = deepcopy(grid)

        @P2G sequential_grid=>i particles=>p weights=>ip begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i] = @∑ w[ip] * m[p] * gravity
            f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
            m⁻¹[i] = inv(m[i]) * !iszero(m[i])
            vⁿ[i] = mv[i] * m⁻¹[i]
            v[i] = vⁿ[i] + (f[i] * m⁻¹[i]) * Δt
        end

        @threaded :static @P2G threaded_grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i] = @∑ w[ip] * m[p] * gravity
            f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
            m⁻¹[i] = inv(m[i]) * !iszero(m[i])
            vⁿ[i] = mv[i] * m⁻¹[i]
            v[i] = vⁿ[i] + (f[i] * m⁻¹[i]) * Δt
        end

        @test threaded_grid.m ≈ sequential_grid.m
        @test threaded_grid.mv ≈ sequential_grid.mv
        @test threaded_grid.f ≈ sequential_grid.f
        @test threaded_grid.v ≈ sequential_grid.v

        sequential_particles = deepcopy(particles)
        threaded_particles = deepcopy(particles)
        α = 0.95

        @G2P sequential_grid=>i sequential_particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * ((1 - α) * v[i] + α * (v[p] + (v[i] - vⁿ[i])))
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt
            F[p] = (one(F[p]) + ∇v[p] * Δt) * F[p]
        end

        @threaded :static @G2P threaded_grid=>i threaded_particles=>p weights=>ip begin
            v[p] = @∑ w[ip] * ((1 - α) * v[i] + α * (v[p] + (v[i] - vⁿ[i])))
            ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            x[p] += @∑ w[ip] * v[i] * Δt
            F[p] = (one(F[p]) + ∇v[p] * Δt) * F[p]
        end

        @test threaded_particles.v ≈ sequential_particles.v
        @test threaded_particles.∇v ≈ sequential_particles.∇v
        @test threaded_particles.x ≈ sequential_particles.x
        @test threaded_particles.F ≈ sequential_particles.F
    end

    @testset "every scheduler matches the sequential partitioned transfer" begin
        grid, particles, weights = transfer_fixture()
        partition = ThreadPartition(grid.x)
        update!(partition, particles.x)

        transfers = (
            :nothing => out -> (@threaded :nothing @P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
            end),
            :static => out -> (@threaded :static @P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
            end),
            :dynamic => out -> (@threaded :dynamic @P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
            end),
            :greedy => out -> (@threaded :greedy @P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
            end),
        )

        # Regions of one color never share a support node, so how the regions
        # are handed out cannot change any node's accumulation order: results
        # must match bit for bit, not just approximately.
        reference = deepcopy(grid)
        last(transfers[1])(reference)
        for (name, transfer!) in transfers[2:end]
            out = deepcopy(grid)
            transfer!(out)
            @test out.m == reference.m
            @test out.mv == reference.mv
            @test out.f == reference.f
        end
    end

    @testset "@P2G hands its assigned fields down to the transfer" begin
        # The fields are passed into `P2G` rather than zeroed ahead of it, which
        # is what lets the threaded path zero them inside the parallel region it
        # already opens instead of paying a fork-join of its own. Zeroing them
        # before the call would still be correct, so this looks at the emitted
        # call rather than at the result.
        function collect_calls(ex, callee, found=Any[])
            Meta.isexpr(ex, :call) && ex.args[1] == callee && push!(found, ex)
            ex isa Expr && foreach(arg -> collect_calls(arg, callee, found), ex.args)
            found
        end

        expanded = @macroexpand @threaded :static @P2G grid=>i particles=>p weights=>ip partition begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i] -= @∑ V[p] * σ[p] * ∇w[ip]
        end
        @test isempty(collect_calls(expanded, :(Tesserae.fillzero!)))
        call = only(collect_calls(expanded, :(Tesserae.P2G)))
        # `-=` accumulates onto what is already there, so `f` is not zeroed.
        @test call.args[end] == :((grid.m, grid.mv))

        # `@G2P2G` zeroes the same way, and passes an empty tuple when its
        # particle half is all it has.
        g2p2g = @macroexpand @threaded :static @G2P2G grid=>i particles=>p weights=>ip partition begin
            v[p] = @∑ w[ip] * v[i]
            m[i] = @∑ w[ip] * m[p]
        end
        @test only(collect_calls(g2p2g, :(Tesserae.G2P2G))).args[end] == :((grid.m,))
        g2p_only = @macroexpand @threaded :static @G2P2G grid=>i particles=>p weights=>ip partition begin
            v[p] = @∑ w[ip] * v[i]
        end
        @test only(collect_calls(g2p_only, :(Tesserae.G2P2G))).args[end] == :(())
    end

    @testset "a threaded transfer zeroes every byte of a grid its particles do not cover" begin
        # The grid is zeroed whole while only the occupied part is scattered
        # into, so a byte the split misses is a byte no later write covers. The
        # other transfer tests use a 3x3 grid, which one worker's chunk covers
        # entirely; this one is big enough for the chunks to have boundaries,
        # with the particles left in a corner so only the zeroing pays for it.
        mesh = CartesianMesh(1.0, (0,400), (0,400))
        GridProp = @NamedTuple begin
            x  :: Vec{2, Float64}
            m  :: Float64
            mv :: Vec{2, Float64}
            f  :: Vec{2, Float64}
        end
        ParticleProp = @NamedTuple begin
            x :: Vec{2, Float64}
            m :: Float64
            V :: Float64
            v :: Vec{2, Float64}
            σ :: SecondOrderTensor{2, Float64, 4}
        end
        grid = generate_grid(GridProp, mesh)
        @test sizeof(grid.m) + sizeof(grid.mv) + sizeof(grid.f) > 4 * Tesserae.FILLZERO_CHUNK_ALIGN * Threads.nthreads()

        particles = generate_particles(ParticleProp, CartesianMesh(1.0, (0,20), (0,20)); alg=GridSampling())
        particles.m .= 1.0
        particles.V .= 1.0
        particles.v .= [rand(Vec{2}) for _ in 1:length(particles)]
        particles.σ .= [rand(SecondOrderTensor{2}) for _ in 1:length(particles)]
        weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
        update!(weights, particles, mesh)
        partition = ThreadPartition(grid.x)
        update!(partition, particles.x)

        # Dirty the fields first, so a byte the zeroing misses shows up as a
        # difference instead of reading as a zero that was there anyway.
        dirty!(g) = (fill!(g.m, NaN); fill!(g.mv, Vec(NaN,NaN)); fill!(g.f, Vec(NaN,NaN)); g)

        reference = dirty!(deepcopy(grid))
        @threaded :nothing @P2G reference=>i particles=>p weights=>ip partition begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i]  = @∑ -V[p] * σ[p] * ∇w[ip]
        end
        @test !any(isnan, reference.m)

        out = dirty!(deepcopy(grid))
        @threaded :dynamic @P2G out=>i particles=>p weights=>ip partition begin
            m[i]  = @∑ w[ip] * m[p]
            mv[i] = @∑ w[ip] * m[p] * v[p]
            f[i]  = @∑ -V[p] * σ[p] * ∇w[ip]
        end
        # Same partition, so the same accumulation order into every node: only
        # the zeroing differs, and it is order-free.
        @test out.m == reference.m
        @test out.mv == reference.mv
        @test out.f == reference.f
    end

    @testset "a failing threaded transfer throws instead of hanging" begin
        grid, particles, weights = transfer_fixture()
        partition = ThreadPartition(grid.x)
        update!(partition, particles.x)

        # A throw from one worker has to release the workers waiting on the
        # phase barrier, otherwise the transfer deadlocks instead of failing.
        # The error must also arrive unwrapped whichever worker raised it.
        @test_throws "boom" (@threaded :static @P2G grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p] * error("boom")
        end)
        @test_throws "boom" (@threaded :dynamic @P2G grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p] * error("boom")
        end)
        @test_throws "boom" (@threaded :greedy @P2G grid=>i particles=>p weights=>ip partition begin
            m[i] = @∑ w[ip] * m[p] * error("boom")
        end)
    end

    # A grid big enough to be split across every worker, so the chunk boundaries
    # are exercised rather than a single worker taking the lot. Few particles
    # keep it quick: that loop walks every node regardless of how many particles
    # there are.
    @testset "threaded grid-node half matches the sequential one (dense)" begin
        mesh = CartesianMesh(0.005, (0,1), (0,1))  # 201^2 = 40401 nodes
        GridProp = @NamedTuple begin
            x::Vec{2,Float64}; m::Float64; m⁻¹::Float64; mv::Vec{2,Float64}; v::Vec{2,Float64}
        end
        grid = generate_grid(GridProp, mesh)
        particles = generate_particles(@NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, grid.x)
        filter!(pt -> all(c -> 0.1 < c < 0.15, pt.x), particles)
        particles.m .= 1.0
        for p in eachindex(particles); particles.v[p] = Vec(0.1, 0.2); end
        weights = generate_basis_weights(BSpline(Quadratic()), grid.x, length(particles))
        update!(weights, particles, grid.x)
        partition = ThreadPartition(mesh)
        update!(partition, particles.x)

        run!(out, schedule) = schedule === :nothing ?
            (@P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                m⁻¹[i] = inv(m[i]) * !iszero(m[i])
                v[i] = mv[i] * m⁻¹[i]
            end) :
            (@threaded @P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                m⁻¹[i] = inv(m[i]) * !iszero(m[i])
                v[i] = mv[i] * m⁻¹[i]
            end)

        reference = deepcopy(grid); run!(reference, :nothing)
        threaded = deepcopy(grid); run!(threaded, :dynamic)
        @test threaded.m == reference.m
        @test threaded.m⁻¹ == reference.m⁻¹
        @test threaded.v == reference.v
    end

    @testset "threaded grid-node half matches the sequential one (SpGrid)" begin
        mesh = CartesianMesh(0.005, (0,1), (0,1))
        GridProp = @NamedTuple begin
            x::Vec{2,Float64}; m::Float64; m⁻¹::Float64; mv::Vec{2,Float64}; v::Vec{2,Float64}
        end
        grid = generate_grid(SpArray, GridProp, mesh)
        particles = generate_particles(@NamedTuple{x::Vec{2,Float64}, m::Float64, v::Vec{2,Float64}}, grid.x)
        filter!(pt -> all(c -> 0.1 < c < 0.15, pt.x), particles)
        particles.m .= 1.0
        for p in eachindex(particles); particles.v[p] = Vec(0.1, 0.2); end
        # Activate every block directly, so the walk has nodes to visit without
        # needing the particles that would otherwise have to fill the mesh.
        update_sparsity!(grid, trues(Tesserae.nblocks(mesh)))
        weights = generate_basis_weights(BSpline(Quadratic()), grid.x, length(particles))
        update!(weights, particles, grid.x)
        partition = ThreadPartition(mesh)
        update!(partition, particles.x)

        run!(out, schedule) = schedule === :nothing ?
            (@P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                m⁻¹[i] = inv(m[i]) * !iszero(m[i])
                v[i] = mv[i] * m⁻¹[i]
            end) :
            (@threaded @P2G out=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
                m⁻¹[i] = inv(m[i]) * !iszero(m[i])
                v[i] = mv[i] * m⁻¹[i]
            end)

        reference = deepcopy(grid); run!(reference, :nothing)
        threaded = deepcopy(grid); run!(threaded, :dynamic)
        for name in (:m, :m⁻¹, :v)
            @test Tesserae.get_data(getproperty(threaded, name)) ==
                  Tesserae.get_data(getproperty(reference, name))
        end
    end

    @testset "threaded P2G requires updated Cartesian partition" begin
        grid, particles, weights = transfer_fixture()
        partition = ThreadPartition(grid.x)
        err = error_message() do
            @threaded :static @P2G grid=>i particles=>p weights=>ip partition begin
                m[i] = @∑ w[ip] * m[p]
            end
        end
        @test occursin("@P2G: No particles assigned to any block in ThreadPartition", err)

        # Each macro must report its own name; @G2P2G and @P2G_Matrix used to
        # borrow @G2P's and @P2G's because they delegated to those checks.
        bogus = collect(1:length(particles))
        for (name, thunk) in ("@P2G"   => () -> (@P2G grid=>i particles=>p bogus=>ip begin
                                                     m[i] = @∑ w[ip]
                                                 end),
                              "@G2P"   => () -> (@G2P grid=>i particles=>p bogus=>ip begin
                                                     v[p] = @∑ w[ip] * v[i]
                                                 end),
                              "@G2P2G" => () -> (@G2P2G grid=>i particles=>p bogus=>ip begin
                                                     ∇v[p] = @∑ v[i] ⊗ v[i]
                                                     m[i]  = @∑ w[ip] * m[p]
                                                 end))
            @test occursin("$name: invalid `BasisWeight`s", error_message(thunk))
        end
    end

    @testset "ordering errors" begin
        ex = quote
            @P2G grid=>i particles=>p weights=>ip begin
                v[i] = mv[i] * m⁻¹[i]
                m[i] = @∑ w[ip] * m[p]
            end
        end
        @test_throws ErrorException macroexpand(@__MODULE__, ex)

        ex = quote
            @G2P2G grid=>i particles=>p weights=>ip begin
                f[i] = @∑ -V[p] * σ[p] * ∇w[ip]
                ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
            end
        end
        @test_throws ErrorException macroexpand(@__MODULE__, ex)
    end

    # The block-scheduled GPU @P2G accumulates a whole grid block in one
    # shared-memory tile, so every particle assigned to a block must have its
    # entire support window inside that block's tile. The kernel writes those
    # slots unchecked, so a basis violating this would corrupt shared memory.
    @testset "block tile contains every support window" begin
        for basis in (BSpline(Constant()), BSpline(Linear()), BSpline(Quadratic()),
                      BSpline(Cubic()), uGIMP(),
                      WLS(BSpline(Quadratic())), KernelCorrection(BSpline(Quadratic())))
            mesh = CartesianMesh(0.02, (0,1), (0,1))
            ParticleProp = @NamedTuple{x::Vec{2,Float64}, l::Float64}
            particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
            particles.l .= 0.01
            weights = generate_basis_weights(basis, mesh, length(particles))
            update!(weights, particles, mesh)
            @test all(eachindex(particles)) do p
                block = Tesserae.findblock(particles.x[p], mesh)
                block === nothing && return true
                Tesserae.p2g_tile_contains(basis, mesh, Tesserae.supportnodes(weights[p]), block)
            end
        end
    end

    @testset "deferred basis weights" begin
        mesh = CartesianMesh(0.02, (0,1), (0,1))
        GridProp = @NamedTuple{x::Vec{2,Float64}, m::Float64, mv::Vec{2,Float64}}
        ParticleProp = @NamedTuple{x::Vec{2,Float64}, l::Float64, m::Float64,
                                   v::Vec{2,Float64}, ∇v::SecondOrderTensor{2,Float64,4}}
        function transfer(weights, particles)
            grid = generate_grid(GridProp, mesh)
            @P2G grid=>i particles=>p weights=>ip begin
                m[i]  = @∑ w[ip] * m[p]
                mv[i] = @∑ w[ip] * m[p] * v[p]
            end
            @G2P grid=>i particles=>p weights=>ip begin
                ∇v[p] = @∑ mv[i] ⊗ ∇w[ip]
            end
            (copy(grid.m), copy(grid.mv), copy(particles.∇v))
        end
        for basis in (BSpline(Linear()), BSpline(Quadratic()), BSpline(Cubic()), uGIMP(),
                      WLS(BSpline(Quadratic())), KernelCorrection(BSpline(Quadratic())),
                      KernelCorrection(uGIMP()))
            particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
            particles.m .= 1.0
            particles.l .= 0.01
            for p in eachindex(particles)
                particles.v[p] = Vec(sin(3particles.x[p][1]), cos(2particles.x[p][2]))
            end
            stored = generate_basis_weights(basis, mesh, length(particles))
            update!(stored, particles, mesh)
            built = generate_basis_weights(basis, mesh, length(particles); deferred=true)
            update!(built, particles, mesh) # no storage to fill, so this is a no-op
            flagged = generate_basis_weights(basis, mesh, length(particles))
            update!(flagged, particles, mesh; deferred=true)

            @test Tesserae.isdeferred(built)     # stores nothing
            @test Tesserae.isdeferred(flagged)   # stores values, told to evaluate
            @test !Tesserae.isdeferred(stored)

            reference = transfer(stored, particles)
            for weights in (built, flagged)
                result = transfer(weights, particles)
                @test result[1] ≈ reference[1]
                @test result[2] ≈ reference[2]
                @test result[3] ≈ reference[3]
            end
            # taking a deferred view must leave the stored values usable
            @test transfer(stored, particles)[1] ≈ reference[1]

            # A fused step writes `x[p]` between the two halves of a `@G2P2G`.
            # Deferred weights must still evaluate at the state the transfer
            # started from, exactly as stored values do, so both halves have to
            # share one per-particle binding taken before the write.
            function fused(weights, particles)
                grid = generate_grid(GridProp, mesh)
                @P2G grid=>i particles=>p weights=>ip begin
                    m[i]  = @∑ w[ip] * m[p]
                    mv[i] = @∑ w[ip] * m[p] * v[p]
                end
                @G2P2G grid=>i particles=>p weights=>ip begin
                    v[p]  = @∑ w[ip] * mv[i] / (m[i] + eps(Float64))
                    x[p]  = x[p] + 0.005 * v[p]
                    m[i]  = @∑ w[ip] * m[p]
                    mv[i] = @∑ w[ip] * m[p] * v[p]
                end
                (copy(grid.m), copy(grid.mv), copy(particles.x))
            end
            fused_reference = fused(stored, deepcopy(particles))
            for weights in (built, flagged)
                moved = fused(weights, deepcopy(particles))
                @test moved[1] ≈ fused_reference[1]
                @test moved[2] ≈ fused_reference[2]
                @test moved[3] ≈ fused_reference[3]
            end

            # A view must keep deferring. It reaches the transfer through the
            # same path stored weights do, and getting this wrong scatters
            # nothing at all rather than raising.
            subset = 2:length(particles)-1
            @test Tesserae.isdeferred(view(built, subset))
            @test Tesserae.isdeferred(view(flagged, subset))
            @test !Tesserae.isdeferred(view(stored, subset))
            let pv = view(particles, subset)
                viewed(w) = begin
                    grid = generate_grid(GridProp, mesh)
                    @P2G grid=>i pv=>p w=>ip begin
                        m[i]  = @∑ w[ip] * m[p]
                        mv[i] = @∑ w[ip] * m[p] * v[p]
                    end
                    (copy(grid.m), copy(grid.mv))
                end
                want = viewed(view(stored, subset))
                @test !iszero(sum(want[1]))
                for weights in (built, flagged)
                    got = viewed(view(weights, subset))
                    @test got[1] ≈ want[1]
                    @test got[2] ≈ want[2]
                end
            end

            # the flag is a per-step choice: clearing it refills and reads again
            @test update!(built, particles, mesh; deferred=true) === built
            @test_throws ErrorException update!(built, particles, mesh; deferred=false)
            update!(flagged, particles, mesh)
            @test !Tesserae.isdeferred(flagged)
            @test transfer(flagged, particles)[1] ≈ reference[1]
        end
        # bases that correct near boundaries read the filter `update!` was given,
        # and must reach the same answer deferred as stored
        @testset "deferred with a boundary filter" begin
            masked = generate_particles(ParticleProp, mesh; alg=GridSampling())
            masked.m .= 1.0
            masked.l .= 0.01
            for p in eachindex(masked)
                masked.v[p] = Vec(sin(3masked.x[p][1]), cos(2masked.x[p][2]))
            end
            mask = falses(size(mesh))
            for basis in (WLS(BSpline(Quadratic())), KernelCorrection(BSpline(Quadratic())),
                          KernelCorrection(uGIMP()))
                fill!(mask, false)
                for p in eachindex(masked), i in Tesserae.supportnodes(basis, Tesserae.LazyRow(masked, p), mesh)
                    mask[i] = true
                end
                mask[cld(size(mesh,1),2), cld(size(mesh,2),2)] = false # a hole inside the body
                stored = generate_basis_weights(basis, mesh, length(masked))
                update!(stored, masked, mesh, mask)
                flagged = generate_basis_weights(basis, mesh, length(masked))
                update!(flagged, masked, mesh, mask; deferred=true)
                expected = transfer(stored, masked)
                actual = transfer(flagged, masked)
                @test actual[1] ≈ expected[1]
                @test actual[2] ≈ expected[2]
                @test actual[3] ≈ expected[3]
            end
        end
        # `@P2G_Matrix` reads the stored values directly instead of going through
        # the transfer's weight resolution, so it must refuse deferred weights
        # rather than assemble a zero matrix from storage that was never filled.
        @testset "deferred weights are refused by @P2G_Matrix" begin
            pts = generate_particles(ParticleProp, mesh; alg=GridSampling())
            pts.m .= 1.0
            grid = generate_grid(GridProp, mesh)
            basis = BSpline(Quadratic())
            assemble(w) = begin
                K = create_sparse_matrix(basis, mesh; ndofs=2)
                @P2G_Matrix grid=>(i,j) pts=>p w=>(ip,jp) begin
                    K[i,j] = @∑ w[ip] * w[jp] * m[p] * one(Mat{2,2,Float64})
                end
                sum(abs, K)
            end
            stored = generate_basis_weights(basis, mesh, length(pts))
            update!(stored, pts, mesh)
            @test assemble(stored) > 0
            flagged = generate_basis_weights(basis, mesh, length(pts))
            update!(flagged, pts, mesh; deferred=true)
            @test_throws ErrorException assemble(flagged)
            @test_throws ErrorException assemble(generate_basis_weights(basis, mesh, length(pts); deferred=true))
        end

        # `update!` takes any array of `BasisWeight`s, but only a `BasisWeightArray`
        # has somewhere to record the choice; the rest must say so rather than
        # reaching for a field they do not have.
        let plain = [BasisWeight(BSpline(Quadratic()), mesh) for _ in 1:4],
            pts = generate_particles(ParticleProp, mesh; alg=GridSampling())
            @test !Tesserae.isdeferred(plain)
            @test_throws ErrorException update!(plain, pts, mesh; deferred=true)
            @test_throws ErrorException update!(plain, pts, mesh, trues(size(mesh)); deferred=true)
        end

        # a basis whose support is not a fixed Cartesian block still cannot defer
        @test_throws ErrorException generate_basis_weights(CPDI(), mesh, 4; deferred=true)
    end
end
