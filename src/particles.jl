abstract type SamplingAlgorithm end

struct GridSampling <: SamplingAlgorithm end

function point_sampling(::GridSampling, l::T, domain::Vararg{Tuple{T, T}, dim}) where {dim, T}
    axis((xmin,xmax)) = (xmin+l/2):l:(xmax)
    vec(CartesianMesh(axis.(domain)...))
end

struct PoissonDiskSampling{RNG} <: SamplingAlgorithm
    rng::RNG
    margin::Real
    parallel::Bool
end
PoissonDiskSampling(; margin=0) = PoissonDiskSampling(Random.default_rng(), margin, true)
PoissonDiskSampling(rng; margin=0, parallel=false) = PoissonDiskSampling(rng, margin, parallel)

# Determine minimum distance between particles for Poisson disk sampling
# so that the number of generated particles is almost the same as the grid sampling.
# This empirical equation is slightly different from a previous work (https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
poisson_disk_sampling_minimum_distance(l::Real, dim::Int) = l/(1.37)^(1/√dim)
function point_sampling(pds::PoissonDiskSampling, l::T, domain::Vararg{Tuple{T, T}, dim}) where {dim, T}
    ϵ = T(pds.margin)
    minmax((xmin,xmax)) = (xmin+ϵ, xmax-ϵ)
    d = poisson_disk_sampling_minimum_distance(l, dim)
    reinterpret(Vec{dim, T}, poisson_disk_sampling(pds.rng, T, d, minmax.(domain)...; pds.parallel))
end

function generate_particles(::Type{ParticleProp}, points::AbstractVector{<: Vec}) where {ParticleProp}
    particles = StructVector{ParticleProp}(undef, length(points))
    fillzero!(particles)
    getx(particles) .= points
    particles
end

"""
    generate_particles([ParticleProp], mesh; spacing=0.5, alg=PoissonDiskSampling())

Generate particles with particle `spacing` by sampling `alg`orithm.
"""
function generate_particles(::Type{ParticleProp}, mesh::CartesianMesh{dim, T}; spacing::Real=0.5, alg::SamplingAlgorithm=PoissonDiskSampling()) where {ParticleProp, dim, T}
    domain = tuple.(Tuple(first(mesh)), Tuple(last(mesh)))
    points = point_sampling(alg, Sequoia.spacing(mesh) * T(spacing), domain...)
    particles = generate_particles(ParticleProp, points)
    _reorder_particles!(particles, mesh)
end

function generate_particles(mesh::CartesianMesh{dim, T}; spacing::Real=0.5, alg::SamplingAlgorithm=PoissonDiskSampling()) where {dim, T}
    generate_particles(@NamedTuple{x::Vec{dim, T}}, mesh; spacing, alg).x
end

function _reorder_particles!(particles::AbstractVector, mesh::CartesianMesh)
    ptsinblks = map(_->Int[], CartesianIndices(blocksize(mesh)))
    for p in eachindex(particles)
        I = whichblock(getx(particles)[p], mesh)
        I === nothing || push!(ptsinblks[I], p)
    end
    reorder_particles!(particles, ptsinblks)
end
