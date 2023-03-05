using PoissonDiskSampling

const Particles = StructVector

function generate_points_regularly(lattice::Lattice, n::Int)
    axes = get_axes(lattice)
    r = spacing(lattice) / 2n
    minmax(ax, r) = (first(ax)+r, last(ax)-r)
    Lattice(2r, minmax.(axes, r)...)
end

function generate_points_randomly(lattice::Lattice, n::Int)
    d = spacing(lattice) / n
    minmaxes = map((min,max)->(min,max), Tuple(first(lattice)), Tuple(last(lattice)))
    points = PoissonDiskSampling.generate(minmaxes...; r = only(unique(d)))
    map(eltype(lattice), points)
end

function generate_particles(
        isindomain::Function,
        ::Type{ParticleState},
        lattice::Lattice{dim};
        n::Int = 2,
        random::Bool = false,
        system::CoordinateSystem = NormalSystem(),
    ) where {ParticleState, dim}
    if random
        allpoints = generate_points_randomly(lattice, n)
    else
        allpoints = generate_points_regularly(lattice, n)
    end
    V = prod(last(lattice) - first(lattice)) / length(allpoints)

    # find points `isindomain`
    mask = broadcast(Base.splat(isindomain), allpoints)

    particles = StructVector{ParticleState}(undef, count(mask))
    fillzero!(particles)

    if :x in propertynames(particles)
        @. particles.x = allpoints[mask]
    end
    if :V in propertynames(particles)
        if dim == 2 && system isa Axisymmetric
            @. particles.V = getindex(particles.x, 1) * V
        else
            @. particles.V = V
        end
    end
    if :l in propertynames(particles)
        if random
            l = V^(1/dim)
        else
            l = spacing(lattice) / n
        end
        particles.l .= l
    end

    reorder_particles!(particles, lattice)
    particles
end

function generate_particles(isindomain::Function, lattice::Lattice{dim, T}; kwargs...) where {dim, T}
    ParticleState = minimum_particle_state(Val(dim), T)
    generate_particles(isindomain, ParticleState, lattice; kwargs...)
end

# for grid
generate_particles(isindomain::Function, ::Type{ParticleState}, grid::Grid; kwargs...) where {ParticleState} = generate_particles(isindomain, ParticleState, grid.x; kwargs...)
generate_particles(isindomain::Function, grid::Grid; kwargs...) = generate_particles(isindomain, grid.x; kwargs...)

function generate_particles(::Type{ParticleState}, particles_old::StructVector) where {ParticleState}
    particles = StructVector{ParticleState}(undef, length(particles_old))
    fillzero!(particles)

    if :x in propertynames(particles)
        particles.x .= particles_old.x
    end
    if :V in propertynames(particles)
        particles.V .= particles_old.V
    end
    if :l in propertynames(particles)
        particles.l .= particles_old.l
    end

    particles
end

function generate_particles(particles_old::StructVector)
    T = eltype(particles_old.x)
    ParticleState = minimum_particle_state(Val(length(T)), eltype(T))
    generate_particles(ParticleState, particles_old)
end

function minimum_particle_state(::Val{dim}, ::Type{T}) where {dim, T}
    @NamedTuple begin
        x::Vec{dim, T}
        V::T
        l::T
    end
end

# reorder_particles!
function _reorder_particles!(particles::Particles, ptsinblks::AbstractArray)
    inds = Vector{Int}(undef, sum(length, ptsinblks))

    cnt = 1
    for blocks in threadsafe_blocks(size(ptsinblks))
        @inbounds for blockindex in blocks
            particleindices = ptsinblks[blockindex]
            for i in eachindex(particleindices)
                inds[cnt] = particleindices[i]
                particleindices[i] = cnt
                cnt += 1
            end
        end
    end

    # keep missing particles aside
    if length(inds) != length(particles) # some points are missing
        missed = particles[setdiff(1:length(particles), inds)]
    end

    # reorder particles
    @inbounds particles[1:length(inds)] .= view(particles, inds)

    # assign missing particles to the end part of `particles`
    if length(inds) != length(particles)
        @inbounds particles[length(inds)+1:end] .= missed
    end

    particles
end

function reorder_particles!(particles::Particles, lattice::Lattice)
    xₚ = particles.x
    ptsinblks = map(_->Int[], lattice)
    @inbounds for p in eachindex(xₚ)
        I = whichblock(lattice, xₚ[p])
        I === nothing || push!(ptsinblks[I], p)
    end
    _reorder_particles!(particles, ptsinblks)
end
