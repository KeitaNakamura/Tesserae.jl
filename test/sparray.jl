@testset "SpIndex" begin
    @test Tesserae.isactive(Tesserae.SpIndex(1, 0)) === false
    @test Tesserae.isactive(Tesserae.SpIndex(1, 1)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(1, 2)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 0)) === false
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 1)) === true
    @test Tesserae.isactive(Tesserae.SpIndex(CartesianIndex(1,1), 2)) === true
    A = rand(3,3)
    @test (@inferred A[Tesserae.SpIndex(1,0)]) === zero(eltype(A))
    @test (@inferred A[Tesserae.SpIndex(1,1)]) === A[1]
    @test (@inferred A[Tesserae.SpIndex(CartesianIndex(1,2),0)]) === zero(eltype(A))
    @test (@inferred A[Tesserae.SpIndex(CartesianIndex(1,2),1)]) === A[1,2]
end

@testset "SpIndices" begin
    spinds = Tesserae.SpIndices(12,20)
    @test IndexStyle(spinds) === IndexCartesian()
    @test size(spinds) === (12,20)
    @test Tesserae.nblocks(spinds) === Tesserae.nblocks((12,20))
    @test !all(Tesserae.isactive, spinds)
    @test !all(i->Tesserae.isactive(spinds,i), eachindex(spinds))
    blkspy = rand(Bool, Tesserae.nblocks(spinds))
    n = update_sparsity!(spinds, blkspy)
    @test n == count(blkspy) * (2^Tesserae.BLOCK_SIZE_LOG2)^2 # `^2` is for dimension
    @test n == Tesserae.countnnz(spinds)
    inds = zeros(Int, size(spinds))
    for I in CartesianIndices(inds)
        blk, i = Tesserae.blocklocal(Tuple(I)...)
        if blkspy[blk...]
            linear_blkindex = LinearIndices(blkspy)[blk...]
            nblks = count(blkspy[1:linear_blkindex])
            blkunit = (2^Tesserae.BLOCK_SIZE_LOG2)^2
            index = (nblks-1) * blkunit + i
            inds[I] = index
        end
    end
    @test map(i->i.spindex, spinds) == inds
    active = collect(Tesserae.activeindices(spinds))
    expected_active = filter(I->Tesserae.isactive(spinds, I), CartesianIndices(spinds))
    @test all(Tesserae.isactive, active)
    @test Set(map(Tesserae.logicalindex, active)) == Set(expected_active)
    @test map(Tesserae.storageindex, active) == sort(map(I->spinds[I].spindex, expected_active))
    @test all(i->1 ≤ Tesserae.storageindex(i) ≤ n, active)
    @test isempty(collect(Tesserae.activeindices(Tesserae.SpIndices(12,20))))

    update_sparsity!(spinds, falses(Tesserae.nblocks(spinds)))
    @test Tesserae.countnnz(spinds) == 0
    @test isempty(collect(Tesserae.activeindices(spinds)))
    @test all(i->!Tesserae.isactive(i), spinds)

    edge_spinds = Tesserae.SpIndices(5, 5)
    edge_n = update_sparsity!(edge_spinds, trues(Tesserae.nblocks(edge_spinds)))
    edge_active = collect(Tesserae.activeindices(edge_spinds))
    @test length(edge_active) == length(edge_spinds)
    @test all(i->1 ≤ Tesserae.storageindex(i) ≤ edge_n, edge_active)
    @test Set(map(Tesserae.logicalindex, edge_active)) == Set(CartesianIndices(edge_spinds))
end

@testset "SpArray" begin
    A = SpArray{Float64}(undef, 12, 20)
    @test IndexStyle(A) === IndexCartesian()
    @test size(A) === (12,20)
    @test !all(i->Tesserae.isactive(A,i), eachindex(A))
    @test all(iszero, A)
    blkspy = rand(Bool, Tesserae.nblocks(A))
    n = update_sparsity!(A, blkspy)
    @test length(Tesserae.get_data(A)) === n
    @test Tesserae.storedindices(A) == eachindex(Tesserae.get_data(A))
    @test Set(map(Tesserae.logicalindex, Tesserae.activeindices(A))) ==
          Set(filter(i->Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->(A[i]=rand(); iszero(A[i])), filter(i->!Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->(a=rand(); A[i]=a; A[i]==a), filter(i->Tesserae.isactive(A,i), eachindex(A)))

    fillzero!(A)
    for i in Tesserae.activeindices(A)
        A[i] = Float64(Tesserae.storageindex(i))
    end
    @test all(i->A[Tesserae.logicalindex(i)] == Float64(Tesserae.storageindex(i)), Tesserae.activeindices(A))
    @test all(i->iszero(A[i]), filter(i->!Tesserae.isactive(A,i), eachindex(A)))

    B = SpArray{Float64}(undef, 12, 20)
    update_sparsity!(B, trues(Tesserae.nblocks(B)))
    B .= 2
    A .= B .+ 1
    @test all(i->A[i] == 3, filter(i->Tesserae.isactive(A,i), eachindex(A)))
    @test all(i->iszero(A[i]), filter(i->!Tesserae.isactive(A,i), eachindex(A)))

    # breadcast is tested on grid part
end

@testset "SpArray @P2G" begin
    mesh = CartesianMesh(1.0, (0,2), (0,2))
    GridProp = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        mv :: Vec{2, Float64}
    end
    ParticleProp = @NamedTuple begin
        x :: Vec{2, Float64}
        m :: Float64
        v :: Vec{2, Float64}
    end

    dense_grid = generate_grid(GridProp, mesh)
    sp_grid = generate_grid(SpArray, GridProp, mesh)
    particles = generate_particles(ParticleProp, mesh; alg=GridSampling())
    weights = generate_basis_weights(BSpline(Linear()), mesh, length(particles))
    update!(weights, particles, mesh)

    for p in eachindex(particles)
        particles.m[p] = 1 + 0.1p
        particles.v[p] = Vec(0.2p, -0.3p)
    end

    update_sparsity!(sp_grid, trues(Tesserae.nblocks(sp_grid)))
    @test all(i->Tesserae.isactive(sp_grid, i), eachindex(sp_grid))
    @test all(x->Tesserae.get_spinds(x) === Tesserae.get_spinds(sp_grid), (sp_grid.m, sp_grid.mv))

    @P2G dense_grid=>i particles=>p weights=>ip begin
        m[i] = @∑ w[ip] * m[p]
        mv[i] = @∑ w[ip] * m[p] * v[p]
    end

    @P2G sp_grid=>i particles=>p weights=>ip begin
        m[i] = @∑ w[ip] * m[p]
        mv[i] = @∑ w[ip] * m[p] * v[p]
    end

    @test sp_grid.m ≈ dense_grid.m
    @test sp_grid.mv ≈ dense_grid.mv
    @test all(!iszero, map(Tesserae.storageindex, Tesserae.activeindices(sp_grid.m)))
end
