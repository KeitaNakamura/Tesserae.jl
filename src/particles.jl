abstract type SamplingAlgorithm end

struct GridSampling <: SamplingAlgorithm end

function point_sampling(::GridSampling, l::T, domain::Vararg{Tuple{T, T}, dim}) where {dim, T}
    axis((xmin,xmax)) = (xmin+l/2):l:(xmax)
    vec(CartesianMesh(axis.(domain)...))
end

struct PoissonDiskSampling{RNG} <: SamplingAlgorithm
    rng::RNG
    multithreading::Bool
end
PoissonDiskSampling() = PoissonDiskSampling(Random.default_rng(), true)
PoissonDiskSampling(rng; multithreading=false) = PoissonDiskSampling(rng, multithreading)

# Determine minimum distance between particles for Poisson disk sampling
# so that the number of generated particles is almost the same as the grid sampling.
# This empirical equation is slightly different from a previous work (https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
poisson_disk_sampling_minimum_distance(l::Real, dim::Int) = l/(1.37)^(1/√dim)
function point_sampling(pds::PoissonDiskSampling, l::T, domain::Vararg{Tuple{T, T}, dim}) where {dim, T}
    d = poisson_disk_sampling_minimum_distance(l, dim)
    reinterpret(Vec{dim, T}, poisson_disk_sampling(pds.rng, T, d, domain...; pds.multithreading))
end

function generate_particles(::Type{ParticleProp}, points::AbstractVector{<: Vec}) where {ParticleProp}
    _generate_particles(ParticleProp, points)
end
function _generate_particles(::Type{ParticleProp}, points::AbstractVector{<: Vec}) where {ParticleProp}
    if !(isbitstype(ParticleProp))
        error("generate_particles: the property type of grid must be `isbitstype` type")
    end
    particles = StructVector{ParticleProp}(undef, length(points))
    fillzero!(particles)
    getx(particles) .= points
    particles
end

"""
    generate_particles([ParticleProp], mesh; spacing=1/2, alg=PoissonDiskSampling())

Generate particles with a particle `spacing` (`0 < spacing ≤ 1`) sampling the `alg`orithm.
When using `GridSampling()`, setting `spacing=1/η` will generate `η^dim` particles per cell,
where `dim` is the problem dimension.
"""
function generate_particles(
        ::Type{ParticleProp}, mesh::CartesianMesh{dim, T};
        spacing::Real=1/2, alg::SamplingAlgorithm=PoissonDiskSampling()) where {ParticleProp, dim, T}
    points = _generate_points(ParticleProp, mesh, spacing, alg)
    particles = _generate_particles(ParticleProp, points)
    _reorder_particles!(particles, mesh)
end

function generate_particles(mesh::CartesianMesh{dim, T}; spacing::Real=1/2, alg::SamplingAlgorithm=PoissonDiskSampling()) where {dim, T}
    generate_particles(@NamedTuple{x::Vec{dim, T}}, mesh; spacing, alg).x
end

function _generate_points(::Type{ParticleProp}, mesh::CartesianMesh{dim, T}, spacing::Real, alg::SamplingAlgorithm) where {ParticleProp, dim, T}
    domain = tuple.(Tuple(get_xmin(mesh)), Tuple(get_xmax(mesh)))
    point_sampling(alg, Tesserae.spacing(mesh) * T(spacing), domain...)
end

function _reorder_particles!(particles::AbstractVector, mesh::CartesianMesh)
    ptsinblks = map(_->Int[], CartesianIndices(blocksize(mesh)))
    for p in eachindex(particles)
        I = whichblock(getx(particles)[p], mesh)
        I === nothing || push!(ptsinblks[I], p)
    end
    reorder_particles!(particles, ptsinblks)
end
