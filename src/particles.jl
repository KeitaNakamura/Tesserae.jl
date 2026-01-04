abstract type SamplingAlgorithm end

"""
    GridSampling(; subdiv = 2, density = 1)

Generate uniformly spaced particles aligned with the background grid.
The particle spacing is set by the integer subdivision factor `subdiv`:

- `l = h / subdiv`, where `h = spacing(mesh)` is the grid cell size.

The per-cell particle count is controlled by `subdiv` and the discrete multiplier `density`:
- `density ∈ (1, 2, 4)`
- Particles per cell: `density * subdiv^dim`, where `dim` is the problem dimension.

# Interpretation
- `subdiv` refines the spacing (smaller `l`).
- `density` increases the number of particles per cell by a factor of 1×, 2×, or 4×
  while keeping the same spacing `l` (it adds additional offset copies of the same uniform grid).

# Limitation
- `density = 4` is defined only for `dim = 2` or `dim = 3`.
"""
@kwdef struct GridSampling <: SamplingAlgorithm
    subdiv  :: Int = 2 # for all axes
    density :: Int = 1 # (1,2,4)
    spacing :: Float64 = 0.0
end

function generate_points(alg::GridSampling, mesh::CartesianMesh{dim, T}) where {dim, T}
    if alg.spacing != 0
        η = inv(alg.spacing)
        subdiv = round(Int, η)
        subdiv ≈ η || error("`GridSampling(spacing=...)` has changed. Use `GridSampling(; subdiv=..., density=...)` instead.")
        @warn "`GridSampling(spacing=...)` has changed. Interpreting `spacing=$(alg.spacing)` as `GridSampling(; subdiv=$subdiv, density=1)`."
        alg = GridSampling(; subdiv)
    end
    @assert alg.subdiv ≥ 1
    @assert alg.density in (1,2,4) "density must be 1, 2, or 4"

    V = Vec{dim, T}
    O = zero(V)

    shifts = if alg.density == 1
        [O]
    elseif alg.density == 2
        [O, O .+ T(0.5)]
    else
        if dim == 3
            [V(0,   0,   0),
             V(0,   0.5, 0.5),
             V(0.5, 0,   0.5),
             V(0.5, 0.5, 0)]
        elseif dim == 2
            [V(0,   0),
             V(0.5, 0),
             V(0,   0.5),
             V(0.5, 0.5)]
        else
            error("density=4 is only defined for dim=2 or dim=3")
        end
    end

    m = alg.subdiv
    h = spacing(mesh)
    xmin = get_xmin(mesh)
    ncell = size(mesh) .- 1

    q = O .+ ifelse(alg.density == 1, T(0.5), T(0.25))
    pts = Vec{dim, T}[]
    sizehint!(pts, prod(ncell) * alg.density * m^dim)
    for I in CartesianIndices(ncell)
        x0 = xmin + h*Vec(Tuple(I).-1)
        for s in shifts
            for J in CartesianIndices(nfill(m, Val(dim)))
                x = x0 + (h/m)*(Vec(Tuple(J).-1) + q + s)
                push!(pts, x)
            end
        end
    end

    pts
end

struct CellSampling{V <: Union{AbstractVector{<: Vec}, Tuple{Vararg{Vec}}}} <: SamplingAlgorithm
    qpts::V
end

function generate_points(alg::CellSampling, mesh::UnstructuredMesh{<: Any, dim, T}) where {dim, T}
    qpts = alg.qpts
    shape = cellshape(mesh)

    points = Matrix{Vec{dim, T}}(undef, length(qpts), ncells(mesh))
    for c in 1:ncells(mesh)
        indices = cellnodeindices(mesh, c)
        x = mesh[indices]
        for (i, qpt) in enumerate(qpts)
            N = value(shape, qpt)
            points[i,c] = sum(N .* x)
        end
    end
    points
end

"""
    PoissonDiskSampling(spacing = 1/2, rng = Random.default_rng())

The particles are generated based on the Poisson disk sampling.
The `spacing` parameter is used to produce a similar number of particles as are generated with [`GridSampling(subdiv = inv(spacing), density = 1)`](@ref).
"""
@kwdef struct PoissonDiskSampling{T, RNG} <: SamplingAlgorithm
    spacing        :: T    = 1/2
    rng            :: RNG  = Random.default_rng()
    multithreading :: Bool = rng isa Random.TaskLocalRNG ? true : false
end

# Determine minimum distance between particles for Poisson disk sampling
# so that the number of generated particles is almost the same as the grid sampling.
# This empirical equation is slightly different from a previous work (https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
poisson_disk_sampling_minimum_distance(l::Real, dim::Int) = l/(1.37)^(1/√dim)
function generate_points(alg::PoissonDiskSampling, mesh::CartesianMesh{dim, T}) where {dim, T}
    l = T(alg.spacing) * spacing(mesh)
    domain = tuple.(Tuple(get_xmin(mesh)), Tuple(get_xmax(mesh)))
    d = poisson_disk_sampling_minimum_distance(l, dim)
    reinterpret(Vec{dim, T}, poisson_disk_sampling(alg.rng, T, d, domain...; alg.multithreading))
end

function _generate_particles(::Type{ParticleProp}, points::AbstractArray{<: Vec}) where {ParticleProp}
    if !(isbitstype(ParticleProp))
        error("generate_particles: the property type of grid must be `isbitstype` type")
    end
    particles = StructArray{ParticleProp}(undef, size(points))
    fillzero!(particles)
    getx(particles) .= points
    particles
end

"""
    generate_particles([ParticleProp], mesh; alg=PoissonDiskSampling())

Generate particles across the entire `mesh` domain based on the selected `alg` algorithm.
See also [`GridSampling`](@ref) and [`PoissonDiskSampling`](@ref).

The generated `particles` is a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl)
where each element is of type `ParticleProp`. The first field of `ParticleProp` is designated for
particle positions.

```jldoctest
julia> ParticleProp = @NamedTuple begin
           x  :: Vec{2, Float64}
           m  :: Float64
           V  :: Float64
           v  :: Vec{2, Float64}
           F  :: SecondOrderTensor{2, Float64, 4}
           σ  :: SymmetricSecondOrderTensor{2, Float64, 3}
       end
@NamedTuple{x::Vec{2, Float64}, m::Float64, V::Float64, v::Vec{2, Float64}, F::Tensor{Tuple{2, 2}, Float64, 2, 4}, σ::SymmetricSecondOrderTensor{2, Float64, 3}}

julia> mesh = CartesianMesh(0.5, (0,3), (0,2));

julia> particles = generate_particles(ParticleProp, mesh; alg=GridSampling());

julia> particles.F[1] = one(eltype(particles.F));

julia> particles[1]
(x = [0.125, 0.125], m = 0.0, V = 0.0, v = [0.0, 0.0], F = [1.0 0.0; 0.0 1.0], σ = [0.0 0.0; 0.0 0.0])

julia> particles.F
96-element Vector{Tensor{Tuple{2, 2}, Float64, 2, 4}}:
 [1.0 0.0; 0.0 1.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 ⋮
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
```
"""
function generate_particles(
        ::Type{ParticleProp}, mesh::CartesianMesh;
        alg::SamplingAlgorithm=PoissonDiskSampling()) where {ParticleProp}
    points = generate_points(alg, mesh)
    particles = _generate_particles(ParticleProp, points)
    particles
    # _reorder_particles!(particles, mesh)
end

function generate_particles(
        ::Type{ParticleProp}, mesh::UnstructuredMesh;
        alg::SamplingAlgorithm=CellSampling(quadpoints(cellshape(mesh)))) where {ParticleProp}
    points = generate_points(alg, mesh)
    particles = _generate_particles(ParticleProp, points)
    particles
    # _reorder_particles!(particles, mesh)
end

function generate_particles(mesh::CartesianMesh{dim, T}; alg::SamplingAlgorithm=PoissonDiskSampling()) where {dim, T}
    generate_particles(@NamedTuple{x::Vec{dim, T}}, mesh; alg).x
end

function _reorder_particles!(particles::AbstractVector, mesh::CartesianMesh)
    ptsinblks = map(_->Int[], CartesianIndices(blocksize(mesh)))
    for p in eachindex(particles)
        I = whichblock(getx(particles)[p], mesh)
        I === nothing || push!(ptsinblks[I], p)
    end
    reorder_particles!(particles, ptsinblks)
end
